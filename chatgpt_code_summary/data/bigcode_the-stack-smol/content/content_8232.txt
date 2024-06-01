# -*- coding: utf-8 -*-
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

# 28 (edge) * 28
IMAGE_SIZE = 28
# 黑白
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
# 过滤器尺寸
CONV1_SIZE = 5
CONV2_DEEP = 64
CONV2_SIZE = 5
# num of Fully connected nodes
FC_SIZE = 512


# def get_weight_variable(shape, regularizer):
#     weights = tf.get_variable(
#         "weight", shape,
#         initializer=tf.truncated_normal_initializer(stddev=0.1))

#     if regularizer != None:
#         tf.add_to_collection("losses", regularizer(weights))  # 这个是自定义集合，不受自动管理
#     return weights


# def inference(input_tensor, regularizer):
#     with tf.variable_scope("layer1"):
#         weights = get_weight_variable(
#             [INPUT_NODE, LAYER1_NODE], regularizer)  # 注意当这行被多次运行时，记得修改 reuse=True
#         biases = tf.get_variable(
#             "biases", [LAYER1_NODE],
#             initializer=tf.constant_initializer(0.0))
#         layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

#     with tf.variable_scope("layer2"):
#         weights = get_weight_variable(
#             [LAYER1_NODE, OUTPUT_NODE], regularizer)  # 注意当这行被多次运行时，记得修改 reuse=True
#         biases = tf.get_variable(
#             "biases", [OUTPUT_NODE],
#             initializer=tf.constant_initializer(0.0))
#         layer2 = tf.matmul(layer1, weights) + biases

#     return layer2


def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(  # 与 tf.Variable() 类似
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],  # x, y, prev-depth, depth
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_biases = tf.get_variable(
            "bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0)
        )

        # 过滤器：边长5，深度32，移动步长1，填充全0 
        conv1 = tf.nn.conv2d(
            input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME'
        )
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # https://www.jianshu.com/p/cff8678de15a
    # 最大池化层：
    with tf.name_scope('layer2-pool1'):
        # 过滤器：边长2，移动步长2，全0填充
        pool1 = tf.nn.max_pool(
            relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
        )

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_biases = tf.get_variable(
            "bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0)
        )

        conv2 = tf.nn.conv2d(
            pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME'
        )
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(
            relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
        )

    # as_list 拉成向量
    pool_shape = pool2.get_shape().as_list()

    # pool_shape[0] 为一个batch中数据的个数; 7*7*64
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    # reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
    reshaped = tf.reshape(pool2, [-1, nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable(
            "weight", [nodes, FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable(
            "bias", [FC_SIZE], 
            initializer=tf.constant_initializer(0.1)
        )

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: 
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable(
            "weight", [FC_SIZE, NUM_LABELS], 
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable(
            "bias", [NUM_LABELS],
            initializer=tf.constant_initializer(0.1)
        )
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit
