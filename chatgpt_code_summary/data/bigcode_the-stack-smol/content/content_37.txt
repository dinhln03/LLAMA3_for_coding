"""
Tensorflow implementation of DeepFM
"""

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from sklearn import metrics
# from yellowfin import YFOptimizer
import os
import sys
import json

"""
关于 X_i 和 X_v

为什么要把训练数据分成两个矩阵？
FM模型需要为每个特征训练一个embedding vector，
在模型计算过程中使用 embedding_lookup + index matrix 可以方便计算。



首先把特征分成两种，一种是不需要one hot(数值类)，一种是需要one hot（枚举类）。
然后定义，one hot 之前的特征称为 field，one hot 之后的特征为 feature。

- X_i 表示 feat_index
- X_v 表示 feat_value

**feat_index**

feat_index 存储的是样本的 field 的"feature索引"，shape=(N,field_size)。
feat_index[i,j]表示的是第i个样本第j个field的 feature_index。
如果当前 field 不需要 one hot，此 field 就只会映射成一个 feature；
如果当前 field 需要 one hot，此 field 就会被映射成多个 feature ，
每个枚举值是一个 feature，其实就是进行 one hot 编码。

比如 feat_index[i,j]=c，表示 第i个样本第j个 field 的对应着第c个feature，
c是 feature_index。
当然如果 field_j 是数值 field，所有样本的j列都是一样的值，因为 field_j 不需要onehot。
如果 field_j 需要one hot，c的值就是其原来的枚举值onehot后映射对应的 feature_index。
feat_index 是给 embedding_lookup是用的。

**feat_value**

feat_value 存储的是样本field的"值"，shape=(N,field_size)。
feat_value[i,j]表示的是第i个样本第j个field的值。
如果当前field 不需要 one hot，feat_value[i,j]就是原始数据值；
如果当前field 需要 one hot，feat_value[i,j]就是常量1；


注意：这里有一个前提条件，就是 one_hot 的 field 变量只能取一个值，一个变量可以有多个取值的情况是不支持的。

"""


class DeepFM(BaseEstimator, TransformerMixin):

    def __init__(self, feature_size, field_size,
                 embedding_size=8, dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 use_fm=True, use_deep=True,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True, threshold=0.5
                 ):
        assert (use_fm or use_deep)
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size  # 259 denote as M, size of the feature dictionary
        self.field_size = field_size  # 39 denote as F, size of the feature fields
        self.embedding_size = embedding_size  # 8 denote as K, size of the feature embedding

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose  # 是否打印参数总量
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better  # 是否值越大越好
        self.train_result, self.valid_result = [], []
        self.sess = None
        self.graph = None
        self._config = None
        self.threshold = threshold

    def _make_config_pack(self):
        self._config = {

            "feature_size": self.feature_size,  # 259 denote as M, size of the feature dictionary
            "field_size ": self.field_size,  # 39 denote as F, size of the feature fields
            "embedding_size ": self.embedding_size,  # 8 denote as K, size of the feature embedding

            "dropout_fm ": self.dropout_fm,
            "deep_layers ": self.deep_layers,
            "dropout_deep ": self.dropout_deep,
            "deep_layers_activation ": self.deep_layers_activation,
            "use_fm ": self.use_fm,
            "use_deep ": self.use_deep,
            "l2_reg ": self.l2_reg,

            "epoch ": self.epoch,
            "batch_size ": self.batch_size,
            "learning_rate ": self.learning_rate,
            "optimizer_type ": self.optimizer_type,

            "batch_norm ": self.batch_norm,
            "batch_norm_decay ": self.batch_norm_decay,

            "verbose ": self.verbose,  # 是否打印参数总量
            "random_seed ": self.random_seed,
            "loss_type": self.loss_type,
            "eval_metric ": self.eval_metric,
            "greater_is_better ": self.greater_is_better,  # 是否值越大越好
        }

        # self.model_path = '%s/deepfm' % (save_path)

        # self._init_graph()

    def init_graph(self):
        if self.sess is not None:
            return
        self.graph = tf.Graph()
        with self.graph.as_default():

            tf1.set_random_seed(self.random_seed)

            self.feat_index = tf1.placeholder(tf.int32, shape=[None, None],
                                              name="feat_index")  # None * F
            self.feat_value = tf1.placeholder(tf.float32, shape=[None, None],
                                              name="feat_value")  # None * F
            self.label = tf1.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
            self.dropout_keep_fm = tf1.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf1.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf1.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()

            # 每一个feature 有一个 embedding
            # feature_embeddings.shape=(self.feature_size, self.embedding_size)

            # feat_index[i,j] 存储的是 第i条样本第j个field 对应的 feature_index
            # 1. 如果 field_j 是非 one hot 特征，则 field_j 不需要拆成多个 feature，
            #   feat_index[:,j] 所有样本行都是同一个值，对应同一个 feature_index。
            # 2. 如果 field_j 是 one hot 特征，则 field_j 需要拆成多个 feature，每个枚举值独立成一个 feature，
            #   此时 feat_index[:,j] 不同行是不同值，其值表示 枚举值Value(field_j) 对应的 feature_index.
            #   比如，第i=3行样本，第j=5个field表示颜色，其值是红色，红色被 onehot成 feature_index=13.则 feat_index[3,5]=13

            # shape=(N样本数量 * field_size * K)
            # N 表示样本的数量
            # K 是嵌入向量的长度,
            # 取出所有样本，每个 feature 的嵌入向量
            # 对于one_hot 的 field，相当于只取出来枚举值对应的 feature_index 的嵌入向量，
            # 相当于每个 field 取一个，最终每条样本嵌入向量的数量还是 field 。
            self.embeddings = tf.nn.embedding_lookup(
                self.weights["feature_embeddings"],  # shape=(self.feature_size, self.embedding_size)
                self.feat_index  # N * field_size
            )
            # shape=(None * F * 1)
            #
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])  # None * F * 1

            # FM部分的公式是 (x_i * x_j)(v_i*v_j)=(x_i*v_i)(x_j*v_j)
            # 这里先把每个特征的向量乘上其特征值。
            self.embeddings = tf.multiply(self.embeddings, feat_value)  # None * F * K

            # ---------- first order term ----------
            # 对于k维，tf.reduce_sum(x,axis=k-1)的结果是对最里面一维所有元素进行求和
            self.y_first_order = tf.nn.embedding_lookup(self.weights["feature_bias"], self.feat_index)  # None * F * 1
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)  # None * F
            self.y_first_order = tf.nn.dropout(self.y_first_order, rate=1 - self.dropout_keep_fm[0])  # None * F

            # ---------- second order term ---------------
            # sum_square part
            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * K
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # square_sum part
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square,
                                                    self.squared_sum_features_emb)  # None * K
            self.y_second_order = tf.nn.dropout(self.y_second_order, rate=1 - self.dropout_keep_fm[1])  # None * K

            # ---------- Deep component ----------
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])  # None * (F*K)
            self.y_deep = tf.nn.dropout(self.y_deep, rate=1 - self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]),
                                     self.weights["bias_%d" % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase,
                                                        scope_bn="bn_%d" % i)  # None * layer[i] * 1
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep,
                                            rate=1 - self.dropout_keep_deep[1 + i])  # dropout at each Deep layer

            # ---------- DeepFM ----------
            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            elif self.use_deep:
                concat_input = self.y_deep
            self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out, name='out')
                self.loss = tf1.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
            # l2 regularization on weights 正则
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.loss += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights["layer_%d" % i])

            # optimizer
            # 这里可以使用现成的ftrl优化损失
            #  optimizer = tf.train.FtrlOptimizer(lr)  # lr: learningRate
            #  gradients = optimizer.compute_gradients(loss)  # cost
            #  train_op = optimizer.apply_gradients(gradients, global_step=global_step)

            if self.optimizer_type == "adam":
                self.optimizer = tf1.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                         epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf1.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                            initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf1.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                    self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf1.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)
            # elif self.optimizer_type == "yellowfin":
            #     self.optimizer = YFOptimizer(learning_rate=self.learning_rate, momentum=0.0).minimize(
            #         self.loss)

            # init
            self.saver = tf1.train.Saver()
            init = tf1.global_variables_initializer()
            self.sess = self._init_session()
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

    def _init_session(self):
        config = tf1.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True  # 根据运行情况分配GPU内存
        return tf1.Session(config=config)

    def _initialize_weights(self):
        weights = dict()  # 定义参数字典

        # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random.normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            # tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # feature_size * K
        weights["feature_bias"] = tf.Variable(
            # tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name="feature_bias")  # feature_size * 1
            tf.random.uniform([self.feature_size, 1], 0.0, 1.0), name="feature_bias")  # feature_size * 1

        # deep layers
        num_layer = len(self.deep_layers)  # 层数
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))  # 正态分布的标准差
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
            dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def get_batch(self, Xi, Xv, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def fit_on_batch(self, Xi, Xv, y):
        feed_dict = {self.feat_index: Xi,
                     self.feat_value: Xv,
                     self.label: y,
                     self.dropout_keep_fm: self.dropout_fm,
                     self.dropout_keep_deep: self.dropout_deep,
                     self.train_phase: True}
        out, loss, opt = self.sess.run((self.out, self.loss, self.optimizer), feed_dict=feed_dict)
        return out, loss

    def fit(self, Xi_train, Xv_train, y_train,
            Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        """
                :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                                 indi_j is the feature index of feature field j of sample i in the training set
                :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                                 vali_j is the feature value of feature field j of sample i in the training set
                                 vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
                :param y_train: label of each sample in the training set
                :param Xi_valid: list of list of feature indices of each sample in the validation set
                :param Xv_valid: list of list of feature values of each sample in the validation set
                :param y_valid: label of each sample in the validation set
                :param early_stopping: perform early stopping or not
                :param refit: refit the model on the train+valid dataset or not
                :return: None
                """
        has_valid = Xv_valid is not None
        Xi_train = Xi_train.copy()
        Xv_train = Xv_train.copy()
        y_train = y_train.copy()

        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                trian_out, train_loss = self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                # print(trian_out, file=sys.stderr)
                if i % 1000 == 0:
                    # print(trian_out, file=sys.stderr)
                    print("epoch:%d batch:%d train_loss=%.4f" % (epoch, i, train_loss), file=sys.stderr)

            # evaluate training and validation datasets
            train_me = self.evaluate(Xi_train, Xv_train, y_train)
            self.train_result.append(train_me)
            if has_valid:
                valid_me = self.evaluate(Xi_valid, Xv_valid, y_valid)
                self.valid_result.append(valid_me)

            if self.verbose > 0 and epoch % self.verbose == 0:
                print("[%d] [train] auc=%.4f acc=%.4f mse=%.4f precision_1=%.4f recall_1=%.4f [%.1f s]"
                      % (epoch + 1,
                         train_me['auc'],
                         train_me['acc'],
                         train_me['mse'],
                         train_me['precision_1'],
                         train_me['recall_1'],
                         time() - t1))

                if has_valid:
                    print(
                        "[%d] [valid] auc=%.4f acc=%.4f mse=%.4f precision_1=%.4f recall_1=%.4f [%.1f s]"
                        % (epoch + 1,
                           valid_me['auc'],
                           valid_me['acc'],
                           valid_me['mse'],
                           valid_me['precision_1'],
                           valid_me['recall_1'],
                           time() - t1))
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
            Xi_train = Xi_train + Xi_valid
            Xv_train = Xv_train + Xv_valid
            y_train = y_train + y_valid
            for epoch in range(100):
                self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
                total_batch = int(len(y_train) / self.batch_size)
                for i in range(total_batch):
                    Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train,
                                                                 self.batch_size, i)
                    self.fit_on_batch(Xi_batch, Xv_batch, y_batch)
                # check
                train_result = self.evaluate(Xi_train, Xv_train, y_train)
                if abs(train_result - best_train_score) < 0.001 or \
                        (self.greater_is_better and train_result > best_train_score) or \
                        ((not self.greater_is_better) and train_result < best_train_score):
                    break

    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                        valid_result[-2] < valid_result[-3] and \
                        valid_result[-3] < valid_result[-4] and \
                        valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] \
                        and valid_result[-2] > valid_result[-3] \
                        and valid_result[-3] > valid_result[-4] \
                        and valid_result[-4] > valid_result[-5]:
                    return True
        return False

    def predict(self, Xi, Xv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        dummy_y = [1] * len(Xi)
        batch_index = 0
        Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         # self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.train_phase: False}
            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            Xi_batch, Xv_batch, y_batch = self.get_batch(Xi, Xv, dummy_y, self.batch_size, batch_index)

        return y_pred

    def evaluate(self, Xi, Xv, y_true):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        size = y_true.shape[0]
        y_pred = self.predict(Xi, Xv)
        error = y_true - y_pred
        mse = (error * error).sum() / size
        y_pred_m = y_pred.copy()

        y_pred_m[y_pred_m >= self.threshold] = 1
        y_pred_m[y_pred_m < self.threshold] = 0

        # accuracy = metrics.accuracy_score(y_true, y_pred_m)
        cm = metrics.confusion_matrix(y_true, y_pred_m, labels=[1, 0])
        # 实际正样本数量
        real_1_count = cm[0, :].sum()
        # 预测为正样本数量
        predict_1_count = cm[:, 0].sum()
        # 正样本 预测正确的数量
        right_1_count = cm[0, 0]
        if predict_1_count == 0:
            precision_1 = 0
        else:
            # 正样本精确率
            precision_1 = right_1_count / predict_1_count
        if real_1_count == 0:
            recall_1 = 0
        else:
            # 正样本召回率
            recall_1 = right_1_count / real_1_count

        return {
            "size": size,
            "acc": (cm[0, 0] + cm[1, 1]) / size,
            # "实际退费人次": cm[0, :].sum(),
            # "预测退费人次": cm[:, 0].sum(),
            # "预测正确人次": cm[0, 0],
            # "预测错误人次": cm[1, 0],
            "precision_1": precision_1,
            "recall_1": recall_1,
            "auc": self.eval_metric(y_true, y_pred),
            "mse": mse
        }

    def save(self, save_path):
        model_prefix = os.path.join(save_path, 'deepfm')
        print("Save model...", save_path, file=sys.stderr)
        self.saver.save(self.sess, model_prefix)
        if self._config is not None:
            config_path = os.path.join(save_path, "config.json")
            with open(config_path, 'w') as fp:
                json.dump(fp)

        print("Save model done.", save_path, file=sys.stderr)

    def load(self, model_path):

        if self.sess is not None:
            self.sess.close()
        if self.graph is not None:
            self.graph = None
        model_prefix = os.path.join(model_path, 'deepfm')
        # self.sess = tf.Session()
        # with tf.Session() as sess:
        # print('load model', file=sys.stderr)
        # t1 = time()
        print("Load model...", model_path, file=sys.stderr)
        self.sess = tf1.Session()
        saver = tf1.train.import_meta_graph(model_prefix + '.meta', clear_devices=True)
        saver.restore(self.sess, model_prefix)
        self.feat_index = tf1.get_default_graph().get_tensor_by_name('feat_index:0')
        self.feat_value = tf1.get_default_graph().get_tensor_by_name('feat_value:0')
        self.dropout_keep_fm = tf1.get_default_graph().get_tensor_by_name('dropout_keep_fm:0')
        self.dropout_keep_deep = tf1.get_default_graph().get_tensor_by_name('dropout_keep_deep:0')
        self.train_phase = tf1.get_default_graph().get_tensor_by_name('train_phase:0')

        self.out = tf1.get_default_graph().get_tensor_by_name('out:0')

        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as fp:
                self._config = json.load(fp)
        else:
            self._config = None

        print("Load model done", model_path, file=sys.stderr)
