#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
@ide: PyCharm
@author: Pedro Silva
@contact: pedroh21.silva@gmail.com
@created: out-10 of 2019
"""
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kback

from tensorflow import keras


class QRSNet(object):

    @classmethod
    def _cnn_net(cls):
        """
        Create the CNN net topology.
        :return keras.Sequential(): CNN topology.
        """
        qrs_detector = keras.Sequential()

        # CONV1
        qrs_detector.add(keras.layers.Conv1D(96, 49, activation=tf.nn.relu, input_shape=(300, 1), strides=1, name='conv1'))

        # POOLING 1
        qrs_detector.add(keras.layers.MaxPool1D(pool_size=2, strides=2, name='pool1'))

        # CONV2
        qrs_detector.add(keras.layers.Conv1D(128, 25, activation=tf.nn.relu, strides=1, name='conv2'))

        # POOLING 2
        qrs_detector.add(keras.layers.MaxPool1D(pool_size=2, strides=2, name='pool2'))

        # CONV3
        qrs_detector.add(keras.layers.Conv1D(256, 9, activation=tf.nn.relu, strides=1, name='conv3'))

        # POOLING 3
        qrs_detector.add(keras.layers.MaxPool1D(pool_size=2, strides=2, name='pool3'))

        # CONV4
        qrs_detector.add(keras.layers.Conv1D(512, 9, activation=tf.nn.relu, strides=1, name='conv4'))

        # POOLING 4
        qrs_detector.add(keras.layers.MaxPool1D(pool_size=2, strides=2, name='pool4'))

        qrs_detector.add(keras.layers.Flatten(data_format=None, name='flatten'))

        # FC1
        qrs_detector.add(keras.layers.Dense(units=4096, activation=tf.nn.relu, name='fc1'))

        # FC2
        qrs_detector.add(keras.layers.Dense(units=4096, activation=tf.nn.relu, name='fc2'))

        # DROP1
        qrs_detector.add(keras.layers.Dropout(rate=0.5, name='drop1'))

        # Classes
        qrs_detector.add(keras.layers.Dense(units=2, name='classes'))

        # SoftMax
        qrs_detector.add(keras.layers.Activation(activation=tf.nn.softmax, name='softmax'))

        return qrs_detector

    @classmethod
    def build(cls, net_type):
        """
        Build the CNN topology.
        :param str net_type: the network type, CNN or LSTM.
        :return keras.Sequential(): CNN topology.
        """
        if net_type == 'cnn':
            qrs_detector = cls._cnn_net()
        else:
            raise NotImplementedError('Only the CNN network was implemented.')

        return qrs_detector

    @classmethod
    def _prepare_data(cls, data_x, input_shape, data_y, number_of_classes, normalize):
        """
        Prepare the data for the training, turning it into a numpy array.
        :param list data_x: data that will be used to train.
        :param tuple input_shape: the input shape that the data must have to be used as training data.
        :param list data_y: the labels related to the data used to train.
        :param int number_of_classes: number of classes of the problem.
        :param bool normalize: if the data should be normalized (True) or not (False).
        :return np.array: the data processed.
        """
        if len(input_shape) == 2:
            data_x = np.asarray(data_x).reshape(-1, input_shape[0], input_shape[1])  # Reshape for CNN -  should work!!
        elif len(input_shape) == 3:
            data_x = np.asarray(data_x).reshape(-1, input_shape[0], input_shape[1], input_shape[2])  # Reshape for CNN -  should work!!
        else:
            raise Exception('Only inputs of two and three dimensions were implemented.')
        if normalize:
            data_x = data_x / np.amax(data_x)
        data_y = keras.utils.to_categorical(data_y).reshape(-1, number_of_classes)

        return data_x, data_y

    @classmethod
    def train(cls, model, train_x, train_y, validation_x, validation_y, number_of_classes, input_shape=(300, 1),
              epochs=10, lr=1e-4, batch_size=4, optimizer=None, loss=None, metrics=None, normalize=False, show_net_info=True):
        """
        Function used to train the model.
        :param keras.Sequential model: model to be trained.
        :param list train_x: data that will be used to train.
        :param list train_y: the labels related to the data used to train.
        :param list validation_x: data that will be used to validate the model trained.
        :param list validation_y: the labels related to the data used to validate the model trained.
        :param int number_of_classes: number of classes of the problem.
        :param tuple input_shape: the input shape that the data must have to be used as training data.
        :param int epochs: total epochs that the model will be trained.
        :param float lr: learning rate used to train.
        :param int batch_size: batch size used to train.
        :param optimizer: which optimizer will be used to train.
        :param str loss: loss function used during the training.
        :param list metrics: metrics used to evaluate the trained model.
        :param bool normalize: if the data should be normalized (True) or not (False).
        :param bool show_net_info: if the network topology should be showed (True) or not (False).
        :return keras.Sequential, dict: model trained and the history of the training process.
        """
        if optimizer is None:
            optimizer = keras.optimizers.SGD(lr=lr, momentum=0.9, decay=1e-4/epochs)
        if loss is None:
            loss = keras.losses.categorical_crossentropy
        if metrics is None:
            metrics = ['acc']
        elif type(metrics) is not list:
            metrics = [metrics]

        # Set optimizer
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        if show_net_info:
            print(model.summary())

        # Prepare data
        train_x, train_y = cls._prepare_data(train_x, input_shape, train_y, number_of_classes, normalize)
        validation_x, validation_y = cls._prepare_data(validation_x, input_shape, validation_y, number_of_classes, normalize)

        kback.set_value(model.optimizer.lr, lr)
        train_history = model.fit(x=train_x, y=train_y, validation_data=(validation_x, validation_y), batch_size=batch_size, epochs=epochs)
        # H = model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs)

        return model, train_history

    @classmethod
    def save_model(cls, model, model_name):
        try:
            model.save(model_name)
        except OSError:
            # serialize model to JSON
            model_json = model.to_json()
            with open(model_name.replace('.h5', '.json'), 'w') as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(model_name)

    @classmethod
    def load_model(cls, model_name):
        if os.path.exists(model_name.replace('.h5', '.json')):
            # load json and create model
            json_file = open(model_name.replace('.h5', '.json'), 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = keras.models.model_from_json(loaded_model_json)

            # load weights into new model
            loaded_model.load_weights(model_name)
            return loaded_model
        else:
            return keras.models.load_model(model_name)
