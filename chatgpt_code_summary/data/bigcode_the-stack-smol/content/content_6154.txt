#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from time import time
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score


class ProductNN(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size, embedding_size=8,
                 deep_layers=None, deep_init_size=50, dropout_deep=None,
                 deep_layer_activation=tf.nn.relu, epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer="adam", batch_norm=0,
                 batch_norm_decay=0.995, verbose=False, random_seed=2016,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 greater_is_better=True, use_inner=True):
        assert loss_type in ["logloss", "mse"],\
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"
        if deep_layers is None:
            deep_layers = [32, 32]
        if dropout_deep is None:
            dropout_deep = [0.5, 0.5, 0.5]

        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size

        self.deep_layers = deep_layers
        self.deep_init_size = deep_init_size
        self.dropout_dep = dropout_deep
        self.deep_layers_activation = deep_layer_activation

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []

        self.use_inner = use_inner
        self._init_graph()

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            # input data,模型输入
            self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                             name='feat_index')
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None],
                                             name='feat_value')
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')

            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None],
                                                    name='dropout_deep_deep')
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')

            # weight initializing,权重初始化
            self.weights = self._initialize_weights()

            # model
            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)

            # linear signal
            linear_output = []
            for i in range(self.deep_init_size):
                linear_output.append(tf.reshape(tf.reduce_sum(
                    tf.multiply(self.embeddings, self.weights['product-linear'][i]),
                    axis=[1, 2]), shape=(-1, 1)))

            self.lz = tf.concat(linear_output, axis=1)

            # quadratic signal
            quadratic_output = []
            if self.use_inner:
                for i in range(self.deep_init_size):
                    theta = tf.multiply(
                        self.embeddings, tf.reshape(self.weights['product-quadratic-inner'][i], (1, -1, 1)))
                    quadratic_output.append(tf.reshape(
                        tf.norm(tf.reduce_sum(theta, axis=1), axis=1), shape=(-1, 1)))
            else:
                embedding_sum = tf.reduce_sum(self.embeddings, axis=1)
                p = tf.matmul(tf.expand_dims(embedding_sum, 2), tf.expand_dims(embedding_sum, 1))
                for i in range(self.deep_init_size):
                    theta = tf.multiply(p, tf.expand_dims(
                        self.weights['product-quadratic-outer'][i], 0))
                    quadratic_output.append(tf.reshape(
                        tf.reduce_sum(theta, axis=[1, 2]), shape=(-1, 1)))

            self.lp = tf.concat(quadratic_output, axis=1)
            self.y_deep = tf.nn.relu(tf.add(tf.add(self.lz, self.lp), self.weights['product-bias']))
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])

            # deep part
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i+1])

            self.out = tf.add(tf.matmul(self.y_deep, self.weights['output']), self.weights['output_bias'])

            # loss,代价函数
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            # optimizer,优化器选择
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                            momentum=0.95).minimize(self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _initialize_weights(self):
        weights = dict()

        # Sparse Features->Dense Embeddings weight initializing
        # one-hot编码后输入到Embedding的权重矩阵初始化
        weights['feature_embeddings'] = tf.Variable(tf.random_normal(
            [self.feature_size, self.embedding_size], 0.0, 0.01), name='feature_embeddings')
        weights['feature_bias'] = tf.Variable(tf.random_normal(
            [self.feature_size, 1], 0.0, 1.0), name='feature_bias')

        # Product Layers
        if self.use_inner:
            weights['product-quadratic-inner'] = tf.Variable(tf.random_normal(
                [self.deep_init_size, self.field_size], 0.0, 0.01))
        else:
            weights['product-quadratic-outer'] = tf.Variable(tf.random_normal(
                [self.deep_init_size, self.embedding_size, self.embedding_size], 0.0, 0.01))

        weights['product-linear'] = tf.Variable(tf.random_normal(
            [self.deep_init_size, self.field_size, self.embedding_size], 0.0, 0.01))
        weights['product-bias'] = tf.Variable(tf.random_normal([self.deep_init_size, ], 0, 0, 1.0))

        # Deep layers weight initializing,Xavier初始化
        num_layer = len(self.deep_layers)
        input_size = self.deep_init_size
        glorot = np.sqrt(2.0/(input_size + self.deep_layers[0]))    # var(w)=2/(nin+nout)

        weights['layer_0'] = tf.Variable(np.random.normal(
            loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights['bias_0'] = tf.Variable(np.random.normal(
            loc=0, scale=glorot, size=(1, self.deep_layers[0])), dtype=np.float32)

        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(np.random.normal(
                loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(np.random.normal(
                loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer
        glorot = np.sqrt(2.0/(input_size + 1))
        weights['output'] = tf.Variable(np.random.normal(
            loc=0, scale=glorot, size=(self.deep_layers[-1], 1)), dtype=np.float32)
        weights['output_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights

    # noinspection PyMethodMayBeStatic
    def get_batch(self, xi, xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return xi[start:end], xv[start:end], [[y_] for y_ in y[start:end]]

    # noinspection PyMethodMayBeStatic
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def predict(self, xi, xv):
        feed_dict = {self.feat_index: xi,
                     self.feat_value: xv,
                     self.dropout_keep_deep: [1.0] * len(self.dropout_dep),
                     self.train_phase: True}
        out = self.sess.run(self.out, feed_dict=feed_dict)

        return out

    def evaluate(self, xi, xv, y):
        y = [[y_] for y_ in y]
        feed_dict = {self.feat_index: xi,
                     self.feat_value: xv,
                     self.label: y,
                     self.dropout_keep_deep: [1.0] * len(self.dropout_dep),
                     self.train_phase: True}
        loss = self.sess.run([self.loss], feed_dict=feed_dict)

        return loss

    def fit_on_batch(self, xi, xv, y):
        feed_dict = {self.feat_index: xi,
                     self.feat_value: xv,
                     self.label: y,
                     self.dropout_keep_deep: self.dropout_dep,
                     self.train_phase: True}
        loss, opt = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def fit(self, xi_train, xv_train, y_train, xi_valid=None, xv_valid=None,
            y_valid=None, early_stopping=False, refit=False):
        """
        :param xi_train: [[ind1_1, ind1_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param xv_train: [[val1_1, val1_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features)
        :param y_train: label of each sample in the training set
        :param xi_valid: list of list of feature indices of each sample in the validation set
        :param xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        """

        has_valid = xv_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            # shuffle the dataset,打乱dataset顺序
            self.shuffle_in_unison_scary(xi_train, xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            # get batch data and fit them,获得batch数据并fit
            for i in range(total_batch):
                xi_batch, xv_batch, y_batch = self.get_batch(xi_train, xv_train,
                                                             y_train, self.batch_size, i)
                self.fit_on_batch(xi_batch, xv_batch, y_batch)

            # evaluate training and validation dataset,评价train/valid dataset
            train_result = self.evaluate(xi_train, xv_train, y_train)
            self.train_result.append(train_result[0])

            if has_valid:
                valid_result = self.evaluate(xi_valid, xv_valid, y_valid)
                self.valid_result.append(valid_result[0])
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-loss=%.4f, valid-loss=%.4f [%.1f s]"
                          % (epoch + 1, train_result[0], valid_result[0], time() - t1))
                else:
                    print("[%d] train-loss=%.4f [%.1f s]"
                          % (epoch + 1, train_result[0], time() - t1))
            if has_valid and early_stopping and self.training_termination(self.valid_result):
                break

        # fit a few more epoch on train+valid until result reaches the best_train_score
        if has_valid and refit:
            if self.greater_is_better:
                best_valid_score = max(self.valid_result)
            else:
                best_valid_score = min(self.valid_result)

            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]
            xi_train = xi_train + xi_valid
            xv_train = xv_train + xv_valid
            y_train = y_train + y_valid
            for epoch in range(100):
                self.shuffle_in_unison_scary(xi_train, xv_train, y_train)
                total_batch = int(len(y_train) / self.batch_size)
                for i in range(total_batch):
                    xi_batch, xv_batch, y_batch = self.get_batch(xi_train, xv_train,
                                                                 y_train, self.batch_size, i)
                    self.fit_on_batch(xi_batch, xv_batch, y_batch)
                # check the model performance
                train_result = self.evaluate(xi_train, xv_train, y_train)
                ckp1 = abs(train_result - best_train_score) < 0.001
                ckp2 = self.greater_is_better and train_result > best_train_score
                ckp3 = (not self.greater_is_better) and train_result < best_train_score
                if ckp1 or ckp2 or ckp3:
                    break

    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] < valid_result[-3] < valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] > valid_result[-3] > valid_result[-4] > valid_result[-5]:
                    return True
        return False
