import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import tensorflow as tf

slim = tf.contrib.slim

def disc_net_64(img1, img2, target_dim, scope="DISC", reuse=False):
    nets_dict = dict()
    nets_dict['input1'] = img1
    nets_dict['input2'] = img2
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(0.00004)):
            with slim.arg_scope([slim.conv2d], weights_initializer=tf.contrib.slim.variance_scaling_initializer(), stride=2, padding='SAME', activation_fn=tf.nn.relu) :
                with slim.arg_scope([slim.fully_connected], biases_initializer=tf.zeros_initializer()):
                    nets_dict['concat'] = tf.concat([nets_dict['input1'], nets_dict['input2']], axis=3)
                    nets_dict['conv2d0'] = slim.conv2d(nets_dict['concat'], 32, [4, 4], scope='conv2d_0')
                    nets_dict['conv2d1'] = slim.conv2d(nets_dict['conv2d0'], 32, [4, 4], scope='conv2d_1')
                    nets_dict['conv2d2'] = slim.conv2d(nets_dict['conv2d1'], 64, [4, 4], scope='conv2d_2')
                    nets_dict['conv2d3'] = slim.conv2d(nets_dict['conv2d2'], 64, [4, 4], scope='conv2d_3')
                    n = tf.reshape(nets_dict['conv2d3'], [-1, 4*4*64])
                    nets_dict['fc0'] = slim.fully_connected(n, 256, activation_fn=tf.nn.relu, scope = "output_fc0")
                    nets_dict['output'] = slim.fully_connected(nets_dict['fc0'], target_dim, activation_fn=None, scope = "output_fc1")
                    return nets_dict
