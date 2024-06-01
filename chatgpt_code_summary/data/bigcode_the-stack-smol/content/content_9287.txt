# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")


class CNNModel(models.BaseModel):
  """CNN model with L2 regularization."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a CNN model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    model_input = tf.reshape(model_input, [-1, 32, 32])
    model_input = tf.expand_dims(model_input, 3)

    net = slim.conv2d(model_input, 3, [
        3, 3], scope='conv1', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))

    net = slim.conv2d(net, 3, [
        4, 4], scope='conv2', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))

    net = slim.conv2d(net, 3, [
        5, 5], scope='conv3', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))

    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = tf.reshape(net, [-1, 16 * 16 * 3])
    output = slim.fully_connected(
        net,
        vocab_size,
        activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}


class ResNetModel(models.BaseModel):
  """ResNet model with L2 regularization."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a ResNet model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    model_input = tf.reshape(model_input, [-1, 32, 32])
    model_input = tf.expand_dims(model_input, 3)

    net = slim.conv2d(model_input, 3, [
        3, 3], scope='conv1', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))

    net = slim.conv2d(net, 3, [
        4, 4], scope='conv2', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))

    net = slim.conv2d(net, 3, [
        5, 5], scope='conv3', activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))

    shortcut = tf.tile(model_input, [1, 1, 1, 3])

    # ResNet blocks
    for i in range(0, 9):
      temp = net + shortcut
      net = slim.conv2d(temp, 3, [
          3, 3], scope='conv%d_1' % (i+1), activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))
      net = slim.conv2d(temp, 3, [
          4, 4], scope='conv%d_2' % (i+1), activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))
      net = slim.conv2d(temp, 3, [
          5, 5], scope='conv%d_3' % (i+1), activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(l2_penalty))
      shortcut = temp

    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = tf.reshape(net, [-1, 16 * 16 * 3])
    output = slim.fully_connected(
        net,
        vocab_size,
        activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}


class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    output = slim.fully_connected(
        model_input,
        vocab_size,
        activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}


class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(
        tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(
        tf.reshape(expert_activations,
                   [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}
