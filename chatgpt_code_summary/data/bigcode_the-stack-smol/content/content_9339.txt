# authors_name = 'Preetham Ganesh'
# project_title = 'Multi Sensor-based Human Activity Recognition using OpenCV and Sensor Fusion'
# email = 'preetham.ganesh2015@gmail.com'


import numpy as np
import os
import pandas as pd
import itertools
import logging
import sklearn.pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ParameterGrid


logging.getLogger('sklearn').setLevel(logging.FATAL)


def list_combinations_generator(modalities: list):
    """Generates combinations for items in the given list.

        Args:
            modalities: List of modalities available in the dataset.

        Returns:
            Combinations of items in the given list.
    """
    modality_combinations = list()

    # Iterates across modalities to generate combinations based on length.
    for length in range(1, len(modalities) + 1):

        # Generate combinations for the current length.
        current_length_combinations = itertools.combinations(modalities, length)

        # Iterates across the generated combinations to convert it into a list.
        for combination in current_length_combinations:
            current_combination_list = list()
            for k in combination:
                current_combination_list.append(k)
            modality_combinations.append(current_combination_list)

    return modality_combinations


def data_combiner(n_actions: int,
                  subject_ids: list,
                  n_takes: int,
                  modalities: list,
                  skeleton_pose_model: str):
    """Combines skeleton point information for all actions, all takes, given list of subject ids and given list of
    modalities.

        Args:
            n_actions: Total number of actions in the original dataset.
            subject_ids: List of subjects in the current set.
            n_takes: Total number of takes in the original dataset.
            modalities: Current combination of modalities.
            skeleton_pose_model: Current skeleton pose model name which will be used to import skeleton point
                                 information.

        Returns:
            A pandas dataframe which contains combined skeleton point information for all actions, all takes, given list
            of subject ids and given list of modalities.
    """
    combined_modality_skeleton_information = pd.DataFrame()

    # Iterates across actions, subject_ids, takes, and modalities to combine skeleton point information.
    for i in range(1, n_actions + 1):
        for j in range(len(subject_ids)):
            for k in range(1, n_takes + 1):
                data_name = 'a{}_s{}_t{}'.format(i, subject_ids[j], k)

                # Iterates across modalities to import skeleton point information file and adds it to
                # combined_modality_skeleton_information. If file not found, it moves on to the next combination.
                try:

                    # Imports 1st modality's skeleton point information for current data_name and skeleton_pose_model.
                    current_data_name_modality_information = pd.read_csv('../data/normalized_data/{}/{}_{}.csv'.format(
                        modalities[0], data_name, skeleton_pose_model))
                except FileNotFoundError:
                    continue

                # Since, length of modalities in each combination is different. Hence, if length of modalities is
                # greater than 1, then the imported skeleton point information for other modalities will be merged to
                # the skeleton point information for the first modality.
                if len(modalities) != 1:
                    for m in range(1, len(modalities)):
                        current_skeleton_point_information = pd.read_csv('../data/normalized_data/{}/{}_{}.csv'.format(
                            modalities[m], data_name, skeleton_pose_model))
                        current_data_name_modality_information = pd.merge(current_data_name_modality_information,
                                                                          current_skeleton_point_information,
                                                                          on='frame', how='outer')

                # Adds data_name to the imported skeleton point information.
                current_data_name_modality_information['data_name'] = [data_name for _ in range(len(
                    current_data_name_modality_information))]

                # Removes frame column from the imported skeleton point information.
                current_data_name_modality_information = current_data_name_modality_information.drop(columns=['frame'])

                # Adds action column to the imported skeleton point information.
                current_data_name_modality_information['action'] = [i for _ in range(len(
                    current_data_name_modality_information))]

                # Appends currently imported & modified skeleton point information to the combined modality skeleton
                # point information
                combined_modality_skeleton_information = combined_modality_skeleton_information.append(
                    current_data_name_modality_information)

    return combined_modality_skeleton_information


def calculate_metrics(actual_values: np.ndarray,
                      predicted_values: np.ndarray):
    """Using actual_values, predicted_values calculates metrics such as accuracy, balanced accuracy, precision, recall,
    and f1 scores.

        Args:
            actual_values: Actual action labels in the dataset
            predicted_values: Action labels predicted by the currently trained model

        Returns:
            Dictionary contains keys as score names and values as scores which are floating point values.
    """
    return {'accuracy_score': round(accuracy_score(actual_values, predicted_values) * 100, 3),
            'balanced_accuracy_score': round(balanced_accuracy_score(actual_values, predicted_values) * 100, 3),
            'precision_score': round(precision_score(actual_values, predicted_values, average='weighted',
                                                     labels=np.unique(predicted_values)) * 100, 3),
            'recall_score': round(recall_score(actual_values, predicted_values, average='weighted',
                                               labels=np.unique(predicted_values)) * 100, 3),
            'f1_score': round(f1_score(actual_values, predicted_values, average='weighted',
                                       labels=np.unique(predicted_values)) * 100, 3)}


def retrieve_hyperparameters(current_model_name: str):
    """Based on the current_model_name returns a list of hyperparameters used for optimizing the model (if necessary).

        Args:
            current_model_name: Name of the model currently expected to be trained

        Returns:
            A dictionary containing the hyperparameter name and the values that will be used to optimize the model
    """
    # For support_vector_classifier, the hyperparameter tuned is kernel.
    if current_model_name == 'support_vector_classifier':
        parameters = {'kernel': ['linear', 'poly', 'rbf']}

    # For decision_tree_classifier, the hyperparameters tuned are criterion, splitter, and max_depth.
    elif current_model_name == 'decision_tree_classifier':
        parameters = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [2, 3, 4, 5, 6, 7]}

    # For random_forest_classifier or extra_trees_classifier, the hyperparameters tuned are n_estimators, criterion, and
    # max_depth
    elif current_model_name == 'random_forest_classifier' or current_model_name == 'extra_trees_classifier':
        parameters = {'n_estimators': [i * 10 for i in range(2, 11, 2)], 'criterion': ['gini', 'entropy'],
                      'max_depth': [2, 3, 4, 5, 6, 7]}

    # For gradient_boosting_classifier, the hyperparameters tuned are loss, n_estimators, criterion, and max_depth.
    elif current_model_name == 'gradient_boosting_classifier':
        parameters = {'max_depth': [2, 3, 4, 5, 6, 7], 'n_estimators': [i * 10 for i in range(2, 11, 2)]}

    # For gaussian_naive_bayes, none of the hyperparameters are tuned.
    else:
        parameters = {'None': ['None']}

    return parameters


def split_data_input_target(skeleton_data: pd.DataFrame):
    """Splits skeleton_data into input and target datasets by filtering / selecting certain columns.

        Args:
            skeleton_data: Train / Validation / Test dataset used to split / filter certain columns.

        Returns:
            A tuple containing 2 numpy ndarrays for the input and target datasets.
    """
    skeleton_data_input = skeleton_data.drop(columns=['data_name', 'action'])
    skeleton_data_target = skeleton_data['action']
    return np.array(skeleton_data_input), np.array(skeleton_data_target)


def video_based_model_testing(test_skeleton_information: pd.DataFrame,
                              current_model: sklearn):
    """Tests performance of the currently trained model on the validation or testing sets, where the performance is
    evaluated per video / file, instead of evaluating per frame.

        Args:
            test_skeleton_information: Pandas dataframe which contains skeleton point information for all actions,
                                       subject_ids, and takes in the validation or testing sets.
            current_model: Scikit-learn model that is currently being trained and tested.

        Returns:
            A tuple contains the target and predicted action for each video in the validation / testing set.
    """
    # Identifies unique data_names in the validation / testing set.
    test_data_names = np.unique(test_skeleton_information['data_name'])
    test_target_data = []
    test_predicted_data = []

    # Iterates across the identified unique data names
    for i in range(len(test_data_names)):

        # Filters skeleton point information for the current data name.
        current_data_name_skeleton_information = test_skeleton_information[test_skeleton_information['data_name'] ==
                                                                           test_data_names[i]]

        # Splits filtered skeleton point information into input and target data.
        test_skeleton_input_data, test_skeleton_target_data = split_data_input_target(
            current_data_name_skeleton_information)

        # Predicts labels for each frame in the filtered skeleton point information.
        test_skeleton_predicted_data = list(current_model.predict(test_skeleton_input_data))

        # Identifies which predicted label has highest count and appends it to the final predicted data. Also, appends
        # target label to the target data.
        test_target_data.append(max(current_data_name_skeleton_information['action']))
        test_predicted_data.append(max(test_skeleton_predicted_data, key=test_skeleton_predicted_data.count))

    return np.array(test_target_data), np.array(test_predicted_data)


def model_training_testing(train_skeleton_information: pd.DataFrame,
                           validation_skeleton_information: pd.DataFrame,
                           test_skeleton_information: pd.DataFrame,
                           current_model_name: str,
                           parameters: dict):
    """Trains and validates model for the current model name and hyperparameters on the train_skeleton_informaiton and
    validation_skeleton_information.

        Args:
            train_skeleton_information: Pandas dataframe which contains skeleton point information for all actions,
                                        subject_ids, and takes in the Training set.
            validation_skeleton_information: Pandas dataframe which contains skeleton point information for all actions,
                                             subject_ids, and takes in the Validation set.
            test_skeleton_information: Pandas dataframe which contains skeleton point information for all actions,
                                       subject_ids, and takes in the Test set.
            current_model_name: Name of the model currently expected to be trained.
            parameters: Current parameter values used for training and validating the model.

        Returns:
            A tuple which contains the training metrics, validation metrics, & test metrics.
    """
    # Based on the current_model_name, the scikit-learn object is initialized using the hyperparameter (if necessary)
    if current_model_name == 'support_vector_classifier':
        model = SVC(kernel=parameters['kernel'])
    elif current_model_name == 'decision_tree_classifier':
        model = DecisionTreeClassifier(criterion=parameters['criterion'], splitter=parameters['splitter'],
                                       max_depth=parameters['max_depth'])
    elif current_model_name == 'random_forest_classifier':
        model = RandomForestClassifier(n_estimators=parameters['n_estimators'], criterion=parameters['criterion'],
                                       max_depth=parameters['max_depth'])
    elif current_model_name == 'extra_trees_classifier':
        model = ExtraTreesClassifier(n_estimators=parameters['n_estimators'], criterion=parameters['criterion'],
                                     max_depth=parameters['max_depth'])
    elif current_model_name == 'gradient_boosting_classifier':
        model = GradientBoostingClassifier(n_estimators=parameters['n_estimators'], max_depth=parameters['max_depth'])
    else:
        model = GaussianNB()

    # Splits Training skeleton information into input and target data.
    train_skeleton_input_data, train_skeleton_target_data = split_data_input_target(train_skeleton_information)

    # Trains the object created for the model using the training input and target.
    model.fit(train_skeleton_input_data, train_skeleton_target_data)

    # Predict video based action labels for training and validation skeleton information data.
    train_skeleton_target_data, train_skeleton_predicted_data = video_based_model_testing(train_skeleton_information,
                                                                                          model)
    validation_skeleton_target_data, validation_skeleton_predicted_data = video_based_model_testing(
        validation_skeleton_information, model)
    test_skeleton_target_data, test_skeleton_predicted_data = video_based_model_testing(test_skeleton_information, model)

    # Calculates metrics for the predicted action labels for the training and testing sets.
    train_metrics = calculate_metrics(train_skeleton_target_data, train_skeleton_predicted_data)
    validation_metrics = calculate_metrics(validation_skeleton_target_data, validation_skeleton_predicted_data)
    test_metrics = calculate_metrics(test_skeleton_target_data, test_skeleton_predicted_data)

    return train_metrics, validation_metrics, test_metrics


def per_combination_results_export(combination_name: str,
                                   data_split: str,
                                   metrics_dataframe: pd.DataFrame):
    """Exports the metrics_dataframe into a CSV format to the mentioned data_split folder. If the folder does not exist,
    then the folder is created.

        Args:
            combination_name: Name of the current combination of modalities and skeleton pose model.
            data_split: Name of the split the subset of the dataset belongs to.
            metrics_dataframe: A dataframe containing the mean of all the metrics for all the hyperparameters & models.

        Returns:
            None.
    """
    directory_path = '{}/{}'.format('../results/combination_results', combination_name)
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    file_path = '{}/{}.csv'.format(directory_path, data_split)
    metrics_dataframe.to_csv(file_path, index=False)


def appends_parameter_metrics_combination(current_model_name: str,
                                          current_combination_name: str,
                                          current_split_metrics: dict,
                                          split_metrics_dataframe: pd.DataFrame):
    """Appends the metrics for the current model and current parameter combination to the main dataframe.

        Args:
            current_model_name: Name of the model currently being trained.
            current_combination_name: Current combination of parameters used for training the model.
            current_split_metrics: Metrics for the current parameter combination for the model.
            split_metrics_dataframe: Pandas dataframe which contains metrics for the current combination of modalities.

        Returns:
            Updated version of the pandas dataframe which contains metrics for the current combination of modalities.
    """
    current_split_metrics['model_names'] = current_model_name
    current_split_metrics['parameters'] = current_combination_name
    split_metrics_dataframe = split_metrics_dataframe.append(current_split_metrics, ignore_index=True)
    return split_metrics_dataframe


def per_combination_model_training_testing(train_subject_ids: list,
                                           validation_subject_ids: list,
                                           test_subject_ids: list,
                                           n_actions: int,
                                           n_takes: int,
                                           current_combination_modalities: list,
                                           skeleton_pose_model: str,
                                           model_names: list):
    """Combines skeleton point information based on modality combination, and subject id group. Trains, validates, and
    tests the list of classifier models. Calculates metrics for each data split, model and parameter combination.

        Args:
            train_subject_ids: List of subject ids in the training set.
            validation_subject_ids: List of subject ids in the validation set.
            test_subject_ids: List of subject ids in the testing set.
            n_actions: Total number of actions in the original dataset.
            n_takes: Total number of takes in the original dataset.
            current_combination_modalities: Current combination of modalities which will be used to import and combine
                                            the dataset.
            skeleton_pose_model: Name of the model currently used for extracting skeleton model.
            model_names: List of ML classifier model names which will used creating the objects.

        Returns:
            None.
    """
    # Combines skeleton point information based on modality combination, and subject id group.
    train_skeleton_information = data_combiner(n_actions, train_subject_ids, n_takes, current_combination_modalities,
                                               skeleton_pose_model)
    validation_skeleton_information = data_combiner(n_actions, validation_subject_ids, n_takes,
                                                    current_combination_modalities, skeleton_pose_model)
    test_skeleton_information = data_combiner(n_actions, test_subject_ids, n_takes, current_combination_modalities,
                                              skeleton_pose_model)

    # Creating empty dataframes for the metrics for current modality combination's training, validation, and testing
    # datasets.
    metrics_features = ['accuracy_score', 'balanced_accuracy_score', 'precision_score', 'recall_score', 'f1_score']
    train_models_parameters_metrics = pd.DataFrame(columns=['model_names', 'parameters'] + metrics_features)
    validation_models_parameters_metrics = pd.DataFrame(columns=['model_names', 'parameters'] + metrics_features)
    test_models_parameters_metrics = pd.DataFrame(columns=['model_names', 'parameters'] + metrics_features)

    combination_name = '_'.join(current_combination_modalities + [skeleton_pose_model])

    # Iterates across model names and parameter grid for training and testing the classification models.
    for i in range(len(model_names)):

        # Retrieves parameters and generates parameter combinations.
        parameters = retrieve_hyperparameters(model_names[i])
        parameters_grid = ParameterGrid(parameters)

        for j in range(len(parameters_grid)):
            current_parameters_grid_name = ', '.join(['{}={}'.format(k, parameters_grid[j][k]) for k in
                                                      parameters_grid[j].keys()])

            # Performs model training and testing. Also, generates metrics for the data splits.
            training_metrics, validation_metrics, test_metrics = model_training_testing(
                train_skeleton_information, validation_skeleton_information, test_skeleton_information, model_names[i],
                parameters_grid[j])

            # Appends current modality's train, validation, and test metrics to the main dataframes.
            train_models_parameters_metrics = appends_parameter_metrics_combination(
                model_names[i], current_parameters_grid_name, training_metrics, train_models_parameters_metrics)
            validation_models_parameters_metrics = appends_parameter_metrics_combination(
                model_names[i], current_parameters_grid_name, validation_metrics, validation_models_parameters_metrics)
            test_models_parameters_metrics = appends_parameter_metrics_combination(
                model_names[i], current_parameters_grid_name, test_metrics, test_models_parameters_metrics)

            if model_names[i] != 'gaussian_naive_bayes':
                print('modality_combination={}, model={}, {} completed successfully.'.format(
                    combination_name, model_names[i], current_parameters_grid_name))
            else:
                print('modality_combination={}, model={} completed successfully.'.format(combination_name,
                                                                                         model_names[i]))

    # Exports main training, validation and testing metrics into CSV files.
    per_combination_results_export('_'.join(current_combination_modalities + [skeleton_pose_model]), 'train_metrics',
                                   train_models_parameters_metrics)
    per_combination_results_export('_'.join(current_combination_modalities + [skeleton_pose_model]),
                                   'validation_metrics', validation_models_parameters_metrics)
    per_combination_results_export('_'.join(current_combination_modalities + [skeleton_pose_model]), 'test_metrics',
                                   test_models_parameters_metrics)


def main():
    print()
    n_actions = 27
    n_subjects = 8
    n_takes = 4
    skeleton_pose_models = ['coco', 'mpi']
    modalities = ['rgb', 'depth', 'inertial']
    model_names = ['gaussian_naive_bayes', 'support_vector_classifier', 'decision_tree_classifier',
                   'random_forest_classifier', 'extra_trees_classifier', 'gradient_boosting_classifier']
    modality_combinations = list_combinations_generator(modalities)
    train_subject_ids = [i for i in range(1, n_subjects - 1)]
    validation_subject_ids = [n_subjects - 1]
    test_subject_ids = [n_subjects]
    for i in range(len(modality_combinations)):
        for j in range(len(skeleton_pose_models)):
            per_combination_model_training_testing(train_subject_ids, validation_subject_ids, test_subject_ids,
                                                   n_actions, n_takes, modality_combinations[i],
                                                   skeleton_pose_models[j], model_names)
            print()


if __name__ == '__main__':
    main()
