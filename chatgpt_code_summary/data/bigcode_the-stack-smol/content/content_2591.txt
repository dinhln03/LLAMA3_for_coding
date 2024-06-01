"""command line interface for mutation_origin"""
import os
import time
import pickle
from collections import defaultdict
import click
from tqdm import tqdm
import pandas
from numpy import log
from numpy.random import seed as np_seed
from scitrack import CachingLogger
from sklearn.model_selection import train_test_split

from mutation_origin.opt import (_seed, _feature_dim, _enu_path,
                                 _germline_path, _output_path, _flank_size,
                                 _train_size, _enu_ratio,
                                 _numreps, _label_col, _proximal, _usegc,
                                 _training_path, _c_values, _penalty_options,
                                 _n_jobs, _classifier_path, _data_path,
                                 _predictions_path, _alpha_options,
                                 _overwrite, _verbose, _class_prior,
                                 _strategy, _score)
from mutation_origin.preprocess import data_to_numeric
from mutation_origin.encoder import (get_scaler, inverse_transform_response,
                                     transform_response)
from mutation_origin.classify import (logistic_regression, one_class_svm,
                                      predict_origin, naive_bayes, xgboost)
from mutation_origin.util import (dump_json, load_predictions,
                                  get_basename, get_classifier_label,
                                  get_enu_germline_sizes, iter_indices,
                                  load_classifier, open_)
from mutation_origin.postprocess import measure_performance


__author__ = "Gavin Huttley"
__copyright__ = "Copyright 2014, Gavin Huttley"
__credits__ = ["Yicheng Zhu", "Cheng Soon Ong", "Gavin Huttley"]
__license__ = "BSD"
__version__ = "0.3"
__maintainer__ = "Gavin Huttley"
__email__ = "Gavin.Huttley@anu.edu.au"
__status__ = "Development"


LOGGER = CachingLogger()


@click.group()
def main():
    """mutori -- for building and applying classifiers of mutation origin"""
    pass


@main.command()
@_seed
@_enu_path
@_germline_path
@_output_path
@_train_size
@_enu_ratio
@_numreps
@_overwrite
def sample_data(enu_path, germline_path, output_path, seed,
                train_size,
                enu_ratio, numreps, overwrite):
    """creates train/test sample data"""
    if seed is None:
        seed = int(time.time())
    LOGGER.log_args()
    LOGGER.log_versions(['sklearn', 'numpy'])

    # set the random number seed
    np_seed(seed)
    start_time = time.time()
    os.makedirs(output_path, exist_ok=True)
    logfile_path = os.path.join(output_path, "logs/data_sampling.log")
    if os.path.exists(logfile_path) and not overwrite:
        click.secho(f"Exists: {logfile_path}! use overwrite to force.",
                    fg='red')
        return

    LOGGER.log_file_path = logfile_path
    LOGGER.input_file(enu_path)
    LOGGER.input_file(germline_path)

    enu = pandas.read_csv(enu_path, sep="\t", header=0)
    germline = pandas.read_csv(germline_path, sep="\t", header=0)
    train_size = train_size // 2
    test_size = train_size
    train_enu_ratio, test_enu_ratio = enu_ratio
    enu_train_size, germ_train_size = get_enu_germline_sizes(train_size,
                                                             train_enu_ratio)
    enu_test_size, germ_test_size = get_enu_germline_sizes(test_size,
                                                           test_enu_ratio)
    assert min(enu_train_size, germ_train_size,
               enu_test_size, germ_test_size) > 0

    if (2 * train_size > enu.shape[0] or
            2 * train_size > germline.shape[0]):
        print(f"ENU data set size: {enu.shape[0]}")
        print(f"Germline data set size: {germline.shape[0]}")
        print(f"Train set size: {train_size}")
        raise ValueError("2 x train size exceeds"
                         " size of training data source(s)")

    for rep in range(numreps):
        test_outpath = os.path.join(output_path, f"test-{rep}.tsv.gz")
        train_outpath = os.path.join(output_path, f"train-{rep}.tsv.gz")
        enu_training, enu_testing = train_test_split(
            enu,
            test_size=enu_test_size,
            train_size=enu_train_size)

        germ_training, germ_testing = train_test_split(
            germline,
            test_size=germ_test_size,
            train_size=germ_train_size)
        if any(map(lambda x: x.shape[0] == 0,
                   [enu_training, enu_testing, germ_training, germ_testing])):
            raise RuntimeError("screw up in creating test/train set")

        # concat the data frames
        testing = pandas.concat([enu_testing, germ_testing])
        training = pandas.concat([enu_training, germ_training])
        # write out, separately, the ENU and Germline data for train and test
        testing.to_csv(test_outpath, index=False,
                       sep="\t", compression='gzip')
        training.to_csv(train_outpath, index=False,
                        sep="\t", compression='gzip')

        LOGGER.output_file(test_outpath)
        LOGGER.output_file(train_outpath)

    duration = time.time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.),
                       label="run duration (minutes)")
    LOGGER.shutdown()


@main.command()
@_training_path
@_output_path
@_label_col
@_seed
@_score
@_flank_size
@_feature_dim
@_proximal
@_usegc
@_c_values
@_penalty_options
@_n_jobs
@_overwrite
@_verbose
def lr_train(training_path, output_path, label_col, seed, scoring,
             flank_size, feature_dim, proximal,
             usegc, c_values, penalty_options, n_jobs, overwrite, verbose):
    """logistic regression training, validation, dumps optimal model"""
    if not seed:
        seed = int(time.time())

    np_seed(seed)
    LOGGER.log_args()
    LOGGER.log_versions(['sklearn', 'numpy'])

    os.makedirs(output_path, exist_ok=True)

    basename = get_basename(training_path)
    outpath = os.path.join(output_path, f"{basename}-classifier-lr.pkl.gz")
    if os.path.exists(outpath) and not overwrite:
        if verbose > 1:
            click.secho(f"Skipping. {outpath} exists. "
                        "use overwrite to force.",
                        fg='green')
        return

    logfile_path = os.path.join(output_path,
                                f"logs/{basename}-training-lr.log")
    LOGGER.log_file_path = logfile_path
    LOGGER.input_file(training_path)

    start_time = time.time()
    _, resp, feat, n_dims, names = data_to_numeric(training_path,
                                                   label_col, flank_size,
                                                   feature_dim, proximal,
                                                   usegc)

    if usegc:
        # we need to scale the data
        scaler = get_scaler(feat)
        feat = scaler.transform(feat)
    classifier = logistic_regression(feat, resp, seed, scoring,
                                     c_values,
                                     penalty_options.split(","), n_jobs)
    betas = dict(zip(names, classifier.best_estimator_.coef_.tolist()[0]))
    result = dict(classifier=classifier.best_estimator_, betas=betas,
                  scoring=scoring)
    result['feature_params'] = dict(feature_dim=feature_dim,
                                    flank_size=flank_size, proximal=proximal,
                                    usegc=usegc)
    if usegc:
        result['scaler'] = scaler

    with open(outpath, 'wb') as clf_file:
        pickle.dump(result, clf_file)

    LOGGER.output_file(outpath)
    duration = time.time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.),
                       label="run duration (minutes)")
    LOGGER.shutdown()


@main.command()
@_training_path
@_output_path
@_label_col
@_seed
@_score
@_flank_size
@_feature_dim
@_proximal
@_usegc
@_alpha_options
@_class_prior
@_n_jobs
@_overwrite
@_verbose
def nb_train(training_path, output_path, label_col, seed, scoring,
             flank_size, feature_dim, proximal,
             usegc, alpha_options, class_prior, n_jobs, overwrite, verbose):
    """Naive Bayes training, validation, dumps optimal model"""
    if not seed:
        seed = int(time.time())

    np_seed(seed)
    LOGGER.log_args()
    LOGGER.log_versions(['sklearn', 'numpy'])
    os.makedirs(output_path, exist_ok=True)

    basename = get_basename(training_path)
    outpath = os.path.join(output_path, f"{basename}-classifier-nb.pkl.gz")
    logfile_path = os.path.join(output_path,
                                f"logs/{basename}-training-nb.log")
    if os.path.exists(outpath) and not overwrite:
        if verbose > 1:
            click.secho(f"Skipping. {outpath} exists. "
                        "use overwrite to force.",
                        fg='green')
        return

    LOGGER.log_file_path = logfile_path
    LOGGER.input_file(training_path)

    start_time = time.time()
    if class_prior is not None:
        class_labels = list(class_prior)
        encoded = transform_response(class_labels)
        ordered = sorted(zip(encoded, class_labels))
        class_prior = [class_prior[l] for _, l in ordered]

    _, resp, feat, n_dims, names = data_to_numeric(training_path,
                                                   label_col, flank_size,
                                                   feature_dim, proximal,
                                                   usegc)

    if usegc:
        # we need to scale the data
        scaler = get_scaler(feat)
        feat = scaler.transform(feat)
    classifier = naive_bayes(feat, resp, seed, alpha_options, scoring,
                             class_prior=class_prior, n_jobs=n_jobs)
    betas = dict(zip(names, classifier.best_estimator_.coef_.tolist()[0]))
    result = dict(classifier=classifier.best_estimator_, betas=betas,
                  scoring=scoring)
    result['feature_params'] = dict(feature_dim=feature_dim,
                                    flank_size=flank_size, proximal=proximal,
                                    usegc=usegc)
    if usegc:
        result['scaler'] = scaler

    with open_(outpath, 'wb') as clf_file:
        pickle.dump(result, clf_file)

    LOGGER.output_file(outpath)
    duration = time.time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.),
                       label="run duration (minutes)")
    LOGGER.shutdown()


@main.command()
@_training_path
@_output_path
@_label_col
@_seed
@_flank_size
@_feature_dim
@_proximal
@_usegc
@_strategy
@_n_jobs
@_overwrite
@_verbose
def xgboost_train(training_path, output_path, label_col, seed,
                  flank_size, feature_dim, proximal,
                  usegc, strategy, n_jobs, overwrite, verbose):
    """Naive Bayes training, validation, dumps optimal model"""
    if not seed:
        seed = int(time.time())

    np_seed(seed)
    LOGGER.log_args()
    LOGGER.log_versions(['sklearn', 'numpy'])
    os.makedirs(output_path, exist_ok=True)

    basename = get_basename(training_path)
    outpath = os.path.join(output_path, f"{basename}-classifier-xgb.pkl.gz")
    logfile_path = os.path.join(output_path,
                                f"logs/{basename}-training-xgb.log")
    if os.path.exists(outpath) and not overwrite:
        if verbose > 1:
            click.secho(f"Skipping. {outpath} exists. "
                        "use overwrite to force.",
                        fg='green')
        return

    LOGGER.log_file_path = logfile_path
    LOGGER.input_file(training_path)
    start_time = time.time()
    _, resp, feat, n_dims, names = data_to_numeric(training_path,
                                                   label_col, flank_size,
                                                   feature_dim, proximal,
                                                   usegc)

    # hacking feature so all -1 > 0
    resp = [v if v > 0 else 0 for v in resp]

    if usegc:
        # we need to scale the data
        scaler = get_scaler(feat)
        feat = scaler.transform(feat)

    classifier = xgboost(feat, resp, seed, strategy, n_jobs, verbose)
    result = dict(classifier=classifier)
    result['feature_params'] = dict(feature_dim=feature_dim,
                                    flank_size=flank_size, proximal=proximal,
                                    usegc=usegc)
    if usegc:
        result['scaler'] = scaler

    with open(outpath, 'wb') as clf_file:
        pickle.dump(result, clf_file)

    LOGGER.output_file(outpath)
    duration = time.time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.),
                       label="run duration (minutes)")
    LOGGER.shutdown()


@main.command()
@_training_path
@_output_path
@_label_col
@_seed
@_flank_size
@_feature_dim
@_proximal
@_usegc
@_overwrite
@_verbose
def ocs_train(training_path, output_path, label_col, seed,
              flank_size, feature_dim, proximal, usegc, overwrite, verbose):
    """one-class svm training for outlier detection"""
    if seed is None:
        seed = int(time.time())
    LOGGER.log_args()
    LOGGER.log_versions(['sklearn', 'numpy'])
    start_time = time.time()
    os.makedirs(output_path, exist_ok=True)

    basename = get_basename(training_path)
    outpath = os.path.join(output_path, f"{basename}-classifier-ocs.pkl.gz")
    logfile_path = os.path.join(output_path,
                                f"logs/{basename}-training-ocs.log")
    if os.path.exists(outpath) and not overwrite:
        if verbose > 1:
            click.secho(f"Skipping. {outpath} exists. "
                        "use overwrite to force.",
                        fg='green')
        return

    LOGGER.log_file_path = logfile_path
    LOGGER.input_file(training_path)

    start_time = time.time()
    _, _, feat, n_dims, names = data_to_numeric(training_path,
                                                label_col, flank_size,
                                                feature_dim, proximal,
                                                usegc=usegc,
                                                one_class='g')

    classifier = one_class_svm(feat, seed)
    result = dict(classifier=classifier)
    result['feature_params'] = dict(feature_dim=feature_dim,
                                    flank_size=flank_size, proximal=proximal,
                                    usegc=usegc)

    with open(outpath, 'wb') as clf_file:
        pickle.dump(result, clf_file)

    LOGGER.output_file(outpath)
    duration = time.time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.),
                       label="run duration (minutes)")
    LOGGER.shutdown()


@main.command()
@_classifier_path
@_data_path
@_output_path
@_label_col
@_class_prior
@_overwrite
@_verbose
def predict(classifier_path, data_path, output_path, label_col, class_prior,
            overwrite, verbose):
    """predict labels for data"""
    LOGGER.log_args()
    LOGGER.log_versions(['sklearn', 'numpy'])
    classifier, feature_params, scaler = load_classifier(classifier_path)
    class_label = get_classifier_label(classifier)
    if class_prior is not None and class_label == 'lr':
        # https://stats.stackexchange.com/questions/117592/logistic-regression-prior-correction-at-test-time
        # based on above and King and Zeng, we adjust the intercept term such
        # that it is incremented by ln(p(1) / p(-1)) where p(1) is the prior
        # of a 1 label, p(-1)=1-p(1)
        class_labels = list(class_prior)
        encoded = transform_response(class_labels)
        ordered = sorted(zip(encoded, class_labels))
        if 'e' in ordered[0]:
            adj = log(class_prior['g'] / class_prior['e'])
        else:
            adj = log(class_prior['e'] / class_prior['g'])

        classifier.intercept_ += adj

    basename_class = get_basename(classifier_path)
    basename_data = get_basename(data_path)
    basename = f"{basename_class}-{basename_data}"
    outpath = os.path.join(
        output_path,
        f"{basename}-predicted-{class_label}.json.gz")
    os.makedirs(output_path, exist_ok=True)
    logfile_path = os.path.join(output_path,
                                f"logs/{basename}-predict-{class_label}.log")
    if os.path.exists(outpath) and not overwrite:
        if verbose > 1:
            click.secho(f"Skipping. {outpath} exists. "
                        "use overwrite to force.",
                        fg='green')
        return

    LOGGER.log_file_path = logfile_path
    LOGGER.input_file(classifier_path)
    LOGGER.input_file(data_path)

    start_time = time.time()
    # if NB, the score func name is different
    if class_label in ("nb", "xgb"):
        classifier.decision_function = classifier.predict_proba

    fulldata = pandas.read_csv(data_path, sep='\t')

    result = {}
    result['feature_params'] = feature_params
    result['classifier_label'] = class_label
    result['classifier_path'] = classifier_path
    result['predictions'] = defaultdict(list)
    total = fulldata.shape[0] // 2000
    pbar = tqdm(iter_indices(
        fulldata.shape[0], block_size=2000), ncols=80, total=total)
    for indices in pbar:
        data = fulldata.iloc[indices]
        ids, resp, feat, n_dims, names = data_to_numeric(data,
                                                         label_col=label_col,
                                                         **feature_params)
        if scaler:
            feat = scaler.transform(feat)

        predictions, scores = predict_origin(classifier, feat)
        if class_label in ("nb", "xgb"):
            # each `score' is the probability of belong to either class
            # reduce to just the first class
            scores = scores[:, 1].tolist()
        elif class_label == 'ocs':
            scores = scores[:, 0].tolist()

        predictions = inverse_transform_response(predictions)
        result['predictions']['varid'].extend(list(ids))
        result['predictions']['predicted'].extend(list(predictions))
        result['predictions']['scores'].extend(list(scores))

    dump_json(outpath, result)
    LOGGER.output_file(outpath)
    duration = time.time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.),
                       label="run duration (minutes)")
    LOGGER.shutdown()


# def performance -> produces summary stats on trained classifiers
# requires input data and the predicted results
@main.command()
@_data_path
@_predictions_path
@_output_path
@_label_col
@_overwrite
@_verbose
def performance(data_path, predictions_path, output_path, label_col,
                overwrite, verbose):
    """produce measures of classifier performance"""
    LOGGER.log_args()
    LOGGER.log_versions(['sklearn', 'numpy'])
    if not (data_path or predictions_path):
        click.secho("Need data sets!", fg="red")
        exit()

    basename_train = get_basename(data_path)
    basename_pred = get_basename(predictions_path)
    basename = f"{basename_train}-{basename_pred}"
    outpath = os.path.join(
        output_path,
        f"{basename}-performance.json.gz")
    logfile_path = os.path.join(output_path,
                                f"logs/{basename}-performance.log")
    if os.path.exists(outpath) and not overwrite:
        if verbose > 1:
            click.secho(f"Skipping. {outpath} exists. "
                        "Use overwrite to force.",
                        fg='green')
        return

    LOGGER.log_file_path = logfile_path

    LOGGER.input_file(data_path)
    LOGGER.input_file(predictions_path)
    orig = pandas.read_csv(data_path, sep="\t")
    predicted, feature_params, classifier_path, label =\
        load_predictions(predictions_path)
    result = measure_performance(orig, predicted,
                                 label_col)
    result["feature_params"] = feature_params
    result["classifier_path"] = classifier_path
    result["classifier_label"] = label
    dump_json(outpath, result)
    LOGGER.shutdown()


if __name__ == "__main__":
    main()
