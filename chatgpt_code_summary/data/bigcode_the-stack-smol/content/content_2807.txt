#!/usr/bin/env python
# -*- coding: utf-8 -*-

#############
## Imports ##
#############

import os
import sys ; sys.path.append("/home/developer/workspace/rklearn-lib")
import tensorflow as tf

from rklearn.tfoo_v1 import BaseModel

#################
## CIFAR10CNN  ##
#################

class CIFAR10CNN(BaseModel):

    ################
    ## __init__() ##
    ################

    def __init__(self, config, logger = None):
        super().__init__(config, logger)

        try:

            # these parameters are sent to the trainer through the model because it is easier
            self.num_epochs = self.config.cifar10_cnn["num_epochs"] 
            self.learning_rate = self.config.cifar10_cnn["learning_rate"]
            
            self.max_to_keep = self.config.cifar10_cnn["max_to_keep"]
            self.checkpoint_dir = self.config.cifar10_cnn["checkpoint_dir"]
            self.model_dir = self.config.cifar10_cnn["model_dir"]

            os.makedirs(self.checkpoint_dir, exist_ok = True)
            os.makedirs(self.model_dir, exist_ok = True)
        
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logger.error("error msg = {}, error type = {}, error file = {}, error line = {}".format(e, exc_type, fname, exc_tb.tb_lineno))

            raise RuntimeError("Error in CIFAR10CNN construction regarding the checkpoints and model directories!")

    ###################
    ## build_model() ##
    ###################

    def build_model(self):
        """
        Build the custom CNN for the CIFAR-10 dataset.
        """

        # The input data holders (cf. shapes after prepa) 
        self.X = tf.compat.v1.placeholder(tf.float32, shape = (None, 
                                                     self.config.data["image_size"], 
                                                     self.config.data["image_size"], 
                                                     self.config.data["num_channels"]), name="X")           # ex. (50000, 32, 32, 3)
        self.y = tf.compat.v1.placeholder(tf.int32, shape = (None, self.config.data["num_categories"]), name="y")     # ex. (50000, 10)  
        self.train = tf.compat.v1.placeholder(tf.bool)

        # The CNN architecture = conv/poo layers + flatten layer + connected layers
        with tf.name_scope("cnn"):

            # a. Create convolution/pooling layers = conv + drop + pool + conv + drop + pool + conv + pool + conv + drop  
            self.conv1 = tf.layers.conv2d(self.X, 
                                          self.config.cifar10_cnn["num_filters"], 
                                          self.config.cifar10_cnn["filter_size"], 
                                          padding='same', activation=tf.nn.relu)
            self.drop1 = tf.layers.dropout(self.conv1, self.config.cifar10_cnn["keep_prob"], training=self.train)
            self.pool1 = tf.layers.max_pooling2d(self.drop1, 2, 2)

            self.conv2 = tf.layers.conv2d(self.pool1, 
                                          self.config.cifar10_cnn["num_filters"], 
                                          self.config.cifar10_cnn["filter_size"],
                                          padding='same', activation=tf.nn.relu)
            self.drop2 = tf.layers.dropout(self.conv2, self.config.cifar10_cnn["keep_prob"], training=self.train)
            self.pool2 = tf.layers.max_pooling2d(self.drop2, 2, 2)
            
            self.conv3 = tf.layers.conv2d(self.pool2, 
                                          self.config.cifar10_cnn["num_filters"],  
                                          self.config.cifar10_cnn["filter_size"],
                                          padding='same', activation=tf.nn.relu)
            self.pool3 = tf.layers.max_pooling2d(self.conv3, 2, 2)
            
            self.conv4 = tf.layers.conv2d(self.pool3, 
                                          self.config.cifar10_cnn["num_filters"], 
                                          self.config.cifar10_cnn["filter_size"],
                                          padding='same', activation=tf.nn.relu)
            self.drop3 = tf.layers.dropout(self.conv4, self.config.cifar10_cnn["keep_prob"], training=self.train)
    
            # b. Flatten input data
            self.flatten = tf.reshape(self.drop3, [-1, self.config.cifar10_cnn["fc1_nb_units"]])

            # Create connected layers: fc1, fc2
            with tf.contrib.framework.arg_scope([tf.contrib.layers.fully_connected], 
                                                normalizer_fn=tf.contrib.layers.batch_norm, 
                                                normalizer_params={"is_training": self.train}):
                self.fc1 = tf.contrib.layers.fully_connected(self.flatten, self.config.cifar10_cnn["fc1_nb_units"])
                self.fc2 = tf.contrib.layers.fully_connected(self.fc1, self.config.data["num_categories"], activation_fn=None)

            # Compute loss
            with tf.name_scope("loss"):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc2, labels=self.y))

            # Optimizer
            with tf.name_scope("training_op"):
                self.training_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            # Perf metrics
            with tf.name_scope("accuracy"):
                prediction = tf.equal(tf.argmax(self.fc2, 1), tf.argmax(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))


