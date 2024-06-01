# -----------------------------------------------------------------------------
# Copyright (c) 2021 Trevor P. Martin. All rights reserved.
# Distributed under the MIT License.
# -----------------------------------------------------------------------------
from Data import encode_data
# from utils import cross_validation
from Models import utils
from Models import build_models

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
import tensorflow as tf
import copy


class CNN01(tf.keras.Model):
    @staticmethod
    def build(rows, columns, channels, classes):
        model = tf.keras.Sequential()
        input_shape = (rows, columns, channels)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3,3),
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(classes, activation="softmax"))
        return model

class CNN02(tf.keras.Model):
    @staticmethod
    def build(rows, columns, classes):
        model = tf.keras.Sequential()
        input_shape = (rows, columns)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=3,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(classes, activation="softmax"))
        return model

class CNN03(tf.keras.Model):
    @staticmethod
    def build(rows, columns, classes):
        model = tf.keras.Sequential()
        input_shape = (rows, columns)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(classes, activation="softmax"))
        return model

class CNN04(tf.keras.Model):
    @staticmethod
    def build(rows, columns, classes):
        model = tf.keras.Sequential()
        input_shape = (rows, columns)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(classes, activation="softmax"))
        return model

class CNN05(tf.keras.Model):
    @staticmethod
    def build(rows, columns, channels, classes):
        model = tf.keras.Sequential()
        input_shape = (rows, columns, channels)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3,3),
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation="relu",
            padding="same"
            )
        )
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(classes, activation="softmax"))
        return model

class DNN01(tf.keras.Model):
    @staticmethod
    def build(rows, columns, units, classes):
        model = tf.keras.Sequential()
        input_shape = (rows, columns)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dense(units=units//2, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dropout(rate=0.15))
        model.add(tf.keras.layers.Dense(units=units//4, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dense(classes, activation="softmax"))
        return model

class DNN02(tf.keras.Model):
    @staticmethod
    def build(rows, columns, units, classes):
        model = tf.keras.Sequential()
        input_shape = (rows, columns)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dropout(rate=0.50))
        model.add(tf.keras.layers.Dense(units=units//2, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dense(classes, activation="softmax"))
        return model

class DNN03(tf.keras.Model):
    @staticmethod
    def build(rows, columns, units, classes):
        model = tf.keras.Sequential()
        input_shape = (rows, columns)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=units*2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        model.add(tf.keras.layers.Dropout(rate=0.50))
        model.add(tf.keras.layers.Dense(classes, activation="softmax"))
        return model

class RNN(tf.keras.Model):
    @staticmethod
    def build(rows, columns, units, classes):
        model = tf.keras.Sequential()
        input_shape = (rows, columns)
        model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        model.add(tf.keras.layers.LSTM(
            units=units,
            activation='tanh',
            return_sequences=True,
            )
        )
        model.add(tf.keras.layers.Dropout(rate=0.20))
        model.add(tf.keras.layers.LSTM(
            units=units//2,
            activation='tanh',
            )
        )
        model.add(tf.keras.layers.Dropout(rate=0.20))
        model.add(tf.keras.layers.Dense(64, activation="relu"))
        model.add(tf.keras.layers.Dense(classes, activation="softmax"))
        return model


def run(datasets,
        splice_sites,
        sub_models,
        save,
        vis,
        iter,
        metrics,
        summary,
        config,
        num_folds,
        bal,
        imbal,
        imbal_t,
        imbal_f,
        batch_size,
        epochs
    ):
    """
    Parameters
    ----------
    dataset: a string {nn269, ce, hs3d} indicating which dataset to use
    splice_site_type: a string {acceptor, donor} indicating which splice
        site to train on
    model_architecture: a string {cnn, dnn, rnn} indicating which model
        architecture to use for training
    save_model: boolean, whether to save the current model
    bal: boolean, whether to balance the dataset
    summary: boolean, whether to print out the model architecture summary
    config: boolean, whether to print out the model's configuration
    visualize: boolean, whether to save a performance graph of the model
    metrics: boolean, whether to print out the evaluation metrics for the model
    num_folds: int (default 10), the number of folds for k-fold cross validation
    epochs: int (default 15), the number of epochs for the chosen model
    batch_size: int (default 32), the model batch size
    model_iter: integer, the iteration of the current model architecture (e.g.
        if this is the third cnn architecture you are testing, use 3)
    """
    # (acceptor row len, donor row len) by dataset
    network_rows = {
        'acceptor':{
            'nn269':90, 'ce':141,
            'hs3d':140, 'hs2':602,
            'ce2':602, 'dm':602,
            'ar':602, 'or':602,
        },

        'donor':{
            'nn269':15, 'ce':141,
            'hs3d':140, 'hs2':602,
            'ce2':602, 'dm':602,
            'ar':602, 'or':602,
        },

    }

    # initialize selected sub models
    to_run = dict(
        [
            (sub_model,{
                'nn269':'', 'ce':'',
                'hs3d':'', 'hs2':'',
                'ce2':'', 'dm':'',
                'ar':'', 'or':''
            }) for sub_model in sub_models
        ]
    )

    # results dictionary
    results = copy.deepcopy(to_run)

    # populate sub models with encoded data
    for sub_model in sub_models:
        for dataset in datasets:
            # encode datasets -> return (acc_x, acc_y, don_x, don_y)
            to_run[sub_model][dataset] = encode_data.encode(dataset, sub_model, bal)

    # get a metrics dictionary
    evals = dict(
        [
            (sub_model, {
                'f1':'', 'precision':'',
                'sensitivity':'', 'specificity':'',
                'recall':'', 'mcc':'',
                'err_rate':''
            }) for sub_model in sub_models
        ]
    )

    # accumulate results from running cross validation
    for sub_model in sub_models:
        for dataset in datasets:
            if to_run[sub_model][dataset] == '':
                pass
            else:
                results[sub_model][dataset] = utils.cross_validation(
                    num_folds,
                    sub_model,
                    splice_sites,
                    dataset,
                    to_run[sub_model][dataset],# encoded data for dataset (ds)
                    network_rows, # donor, acceptor rows for ds
                    evals,
                    summary,
                    config,
                    batch_size,
                    epochs,
                    save,
                )
                # if vis:

    print(results)
    return results


    # plot results


        # loss_acc_sub_models(
        #     results,
        #     datasets,
        #     sub_models,
        #     epochs,
        #     num_folds,
        #     bal
        # )






    # # different by splice site type
    # if splice_site_type == 'acceptor':
    #     cnn_X_train, cnn_y_train = cnn_acc_x, acc_y
    #     # same name to preserve for loop structure
    #     X_train, y_train = rd_acc_x, acc_y
    #     dataset_row_num = network_rows[dataset][0]
    # if splice_site_type == 'donor':
    #     cnn_X_train, cnn_y_train = cnn_don_x, don_y
    #     X_train, y_train = rd_don_x, don_y
    #     dataset_row_num = network_rows[dataset][1]
    #
    #
    # # if tune_rnn:
    # #     tune_rnn()
    #
    # # perform cross validation
    # # general
    # trn_fold_accs, trn_fold_losses = [], []
    # val_fold_accs, val_fold_losses  = [], []
    # # esplice
    # rnn_va, rnn_vl, cnn_vl, cnn_va, dnn_vl, dnn_va = [],[],[],[],[],[]
    # rnn_ta, rnn_tl, cnn_tl, cnn_ta, dnn_tl, dnn_ta = [],[],[],[],[],[]
    #
    # # this loop inspired by https://www.machinecurve.com/
    # #index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
    # k_fold = KFold(n_splits=num_folds, shuffle=False)
    # fold = 1
    # for train, test in k_fold.split(X_train, y_train):
    #     if model_architecture != 'esplice':
    #         X_trn, y_trn = X_train[train], y_train[train]
    #         X_val, y_val = X_train[test], y_train[test]
    #         if model_architecture=='cnn':
    #             history, model = build_cnn(
    #                 dataset_row_num,
    #                 summary,
    #                 X_trn,
    #                 y_trn,
    #                 batch_size,
    #                 epochs,
    #                 X_val,#becomes X_val
    #                 y_val,#becomes y_val
    #                 fold,
    #                 num_folds
    #             )
    #         if model_architecture=='dnn':
    #             history, model = build_dnn(
    #                 dataset_row_num,
    #                 summary,
    #                 X_trn,
    #                 y_trn,
    #                 batch_size,
    #                 epochs,
    #                 X_val,#becomes X_val
    #                 y_val,#becomes y_val
    #                 fold,
    #                 num_folds
    #             )
    #         if model_architecture=='rnn':
    #             history, model = build_rnn(
    #                 dataset_row_num,
    #                 summary,
    #                 X_trn,
    #                 y_trn,
    #                 batch_size,
    #                 epochs,
    #                 X_val,#becomes X_val
    #                 y_val,#becomes y_val
    #                 fold,
    #                 num_folds
    #             )
    #             # model.predict(X_trn)
    #         val_fold_accs.append(history.history['val_accuracy'])
    #         val_fold_losses.append(history.history['val_loss'])
    #         trn_fold_accs.append(history.history['accuracy'])
    #         trn_fold_losses.append(history.history['loss'])
    #         fold += 1
    #     else:
    #         # set up submodel datasets
    #         cnn_X_trn, cnn_y_trn = cnn_X_train[train], cnn_y_train[train]
    #         cnn_X_val, cnn_y_val = cnn_X_train[test], cnn_y_train[test]
    #         rd_X_trn, rd_y_trn = X_train[train], y_train[train]
    #         rd_X_val, rd_y_val = X_train[test], y_train[test]
    #         # build each submodel
    #         hist01, submodel_01 = build_cnn(
    #             dataset_row_num,
    #             summary,
    #             cnn_X_trn,
    #             cnn_y_trn,
    #             batch_size,
    #             epochs,
    #             cnn_X_val,
    #             cnn_y_val,
    #             fold,
    #             num_folds
    #         )
    #         hist02, submodel_02 = build_dnn(
    #             dataset_row_num,
    #             summary,
    #             rd_X_trn,
    #             rd_y_trn,
    #             batch_size,
    #             epochs,
    #             rd_X_val,
    #             rd_y_val,
    #             fold,
    #             num_folds
    #         )
    #         # hist03, submodel_03 = build_rnn(
    #         #     dataset_row_num,
    #         #     summary,
    #         #     rd_X_trn,
    #         #     rd_y_trn,
    #         #     batch_size,
    #         #     epochs,
    #         #     rd_X_val,
    #         #     rd_y_val,
    #         #     fold,
    #         #     num_folds
    #         # )
    #         models = [submodel_01, submodel_02]#, submodel_03]
    #         trn_scores, val_scores = EnsembleSplice.build(
    #             models,
    #             batch_size,
    #             cnn_X_trn,
    #             cnn_y_trn,
    #             cnn_X_val,
    #             cnn_y_val,
    #             rd_X_trn,
    #             rd_y_trn,
    #             rd_X_val,
    #             rd_y_val,
    #         )
    #         # get final epoch accuracy
    #         trn_fold_accs.append(trn_scores)
    #         val_fold_accs.append(val_scores)
    #         # rnn_va.append(hist03.history['val_accuracy'])
    #         # rnn_vl.append(hist03.history['val_loss'])
    #         # rnn_ta.append(hist03.history['accuracy'])
    #         # rnn_tl.append(hist03.history['loss'])
    #         # cnn_vl.append(hist01.history['val_loss'])
    #         # cnn_va.append(hist01.history['val_accuracy'])
    #         # cnn_tl.append(hist01.history['loss'])
    #         # cnn_ta.append(hist01.history['accuracy'])
    #         # dnn_vl.append(hist02.history['val_loss'])
    #         # dnn_va.append(hist02.history['val_accuracy'])
    #         # dnn_tl.append(hist02.history['loss'])
    #         # dnn_ta.append(hist02.history['accuracy'])
    #
    #         # rnn_va.append(hist03.history['val_accuracy'][-1])
    #         # rnn_vl.append(hist03.history['val_loss'][-1])
    #         # rnn_ta.append(hist03.history['accuracy'][-1])
    #         # rnn_tl.append(hist03.history['loss'][-1])
    #         cnn_vl.append(hist01.history['val_loss'][-1])
    #         cnn_va.append(hist01.history['val_accuracy'][-1])
    #         cnn_tl.append(hist01.history['loss'][-1])
    #         cnn_ta.append(hist01.history['accuracy'][-1])
    #         dnn_vl.append(hist02.history['val_loss'][-1])
    #         dnn_va.append(hist02.history['val_accuracy'][-1])
    #         dnn_tl.append(hist02.history['loss'][-1])
    #         dnn_ta.append(hist02.history['accuracy'][-1])
    #
    #         fold += 1
    #
    #     # do something with predicted values and real values to get AUC-ROC scores
    #     # sklearn.metrics.roc_auc_score
    #     # also get f-score and other scores here
    #     # maybe connect tune_rnn and build_rnn -> get tuned parameters and plug them
    #     # in automatically to RNN
    #
    # if model_architecture != 'esplice':
    #
    #     val_acc_by_epoch = np.apply_along_axis(lambda row: np.mean(row), 1, np.asarray(val_fold_accs).T)
    #     val_loss_by_epoch = np.apply_along_axis(lambda row: np.mean(row), 1, np.asarray(val_fold_losses).T)
    #     trn_acc_by_epoch = np.apply_along_axis(lambda row: np.mean(row), 1, np.asarray(trn_fold_accs).T)
    #     trn_loss_by_epoch = np.apply_along_axis(lambda row: np.mean(row), 1, np.asarray(trn_fold_losses).T)
    #
    #     std_val_acc = np.apply_along_axis(lambda row: np.std(row), 1, np.asarray(val_fold_accs).T)
    #     std_val_loss = np.apply_along_axis(lambda row: np.std(row), 1, np.asarray(val_fold_losses).T)
    #     std_trn_acc = np.apply_along_axis(lambda row: np.std(row), 1, np.asarray(trn_fold_accs).T)
    #     std_trn_loss = np.apply_along_axis(lambda row: np.std(row), 1, np.asarray(trn_fold_losses).T)
    #
    #     values = [
    #         val_acc_by_epoch,
    #         std_val_acc,
    #         trn_acc_by_epoch,
    #         std_trn_acc,
    #         val_loss_by_epoch,
    #         std_val_loss,
    #         trn_loss_by_epoch,
    #         std_trn_loss
    #     ]
    #
    # if model_architecture == 'esplice':
    #
    #     # make a DICTIONARY AREY
    #     # ES_Val_ACc: (vacc, std_va)
    #     mean_good =  lambda seq: np.apply_along_axis(lambda row: np.mean(row), 1, np.asarray(seq).T)
    #     std_good = lambda seq: np.apply_along_axis(lambda row: np.std(row), 1, np.asarray(seq).T)
    #     vacc = val_fold_accs
    #     tacc = trn_fold_accs
    #     # std_va = val_fold_accs
    #     # std_ta = trn_fold_accs
    #
    #     values = [
    #         val_fold_accs,
    #         trn_fold_accs,
    #         #rnn_va,
    #         # rnn_vl,
    #         #rnn_ta,
    #         # rnn_tl,
    #         # cnn_vl,
    #         cnn_va,
    #         # cnn_tl,
    #         cnn_ta,
    #         # dnn_vl,
    #         dnn_va,
    #         # dnn_tl,
    #         dnn_ta
    #     ]
    #
    #     # cnn_mva = mean_good(cnn_va)
    #     # cnn_mvl = mean_good(cnn_vl)
    #     # cnn_mta = mean_good(cnn_ta)
    #     # cnn_mtl = mean_good(cnn_tl)
    #     # cnn_sva = std_good(cnn_va)
    #     # cnn_svl = std_good(cnn_vl)
    #     # cnn_sta = std_good(cnn_ta)
    #     # cnn_stl = std_good(cnn_tl)
    #     #
    #     # dnn_mva = mean_good(dnn_va)
    #     # dnn_mvl = mean_good(dnn_vl)
    #     # dnn_mta = mean_good(dnn_ta)
    #     # dnn_mtl = mean_good(dnn_tl)
    #     # dnn_sva = std_good(dnn_va)
    #     # dnn_svl = std_good(dnn_vl)
    #     # dnn_sta = std_good(dnn_ta)
    #     # dnn_stl = std_good(dnn_tl)
    #     #
    #     # rnn_mva = mean_good(rnn_va)
    #     # rnn_mvl = mean_good(rnn_vl)
    #     # rnn_mta = mean_good(rnn_ta)
    #     # rnn_mtl = mean_good(rnn_tl)
    #     # rnn_sva = std_good(rnn_va)
    #     # rnn_svl = std_good(rnn_vl)
    #     # rnn_sta = std_good(rnn_ta)
    #     # rnn_stl = std_good(rnn_tl)
    #
    #     # values = [
    #     #     vacc,
    #     #     # std_va,
    #     #     tacc,
    #     #     # std_ta,
    #     #     cnn_mva,
    #     #     cnn_sva,
    #     #     cnn_mvl,
    #     #     cnn_svl,
    #     #     cnn_mta,
    #     #     cnn_sta,
    #     #     cnn_mtl,
    #     #     cnn_stl,
    #     #     dnn_mva,
    #     #     dnn_sva,
    #     #     dnn_mvl,
    #     #     dnn_svl,
    #     #     dnn_mta,
    #     #     dnn_sta,
    #     #     dnn_mtl,
    #     #     dnn_stl,
    #     #     rnn_mva,
    #     #     rnn_sva,
    #     #     rnn_mvl,
    #     #     rnn_svl,
    #     #     rnn_mta,
    #     #     rnn_sta,
    #     #     rnn_mtl,
    #     #     rnn_stl,
    #     # ]

    # if config:
    #     print(model.get_config())
    # if save_model:
    #     name = input('What would you like to name this model?: ')
    #     model.save(f'{name}')
    #     tf.keras.utils.plot_model(model, f'{name}.png', show_shapes=True)
    # if visualize:
    #     loss_acc_esplice(
    #         values,
    #         model_architecture,
    #         dataset,
    #         splice_site_type,
    #         num_folds,
    #         epochs,
    #         bal,
    #     )
