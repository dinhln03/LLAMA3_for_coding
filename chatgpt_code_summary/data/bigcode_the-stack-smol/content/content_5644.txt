import tensorflow as tf
from tensorflow.keras.models import Model
import pandas as pd
import matplotlib.pyplot as plt

import os
import logging

from .common import create_directories


def get_prepared_model(stage: str, no_classes: int, input_shape: list, loss: str, optimizer: str, metrics: list) -> \
        Model:
    """Function creates ANN model and compile.
    Args:
        stage ([str]): stage of experiment
        no_classes ([INT]): No of classes for classification
        input_shape ([int, int]): Input shape for model's input layer
        loss ([str]): Loss function for model
        optimizer ([str]): Optimizer for model
        metrics ([str]): Metrics to watch while training
    Returns:
        model: ANN demo model
    """
    # Define layers
    LAYERS = []
    BASE_LAYERS = [
        tf.keras.layers.Flatten(input_shape=input_shape, name='input_layer'),
        tf.keras.layers.Dense(units=392, activation='relu', name='hidden1'),
        tf.keras.layers.Dense(units=196, activation='relu', name='hidden2'),
        tf.keras.layers.Dense(units=no_classes, activation='softmax', name='output_layer')
    ]

    KERNEL_INIT_LAYERS = [
        tf.keras.layers.Flatten(input_shape=input_shape, name='input_layer'),
        tf.keras.layers.Dense(units=392, activation='relu', name='hidden1', kernel_initializer='glorot_uniform',
                              bias_initializer='zeros'),
        tf.keras.layers.Dense(units=196, activation='relu', name='hidden2', kernel_initializer='glorot_uniform',
                              bias_initializer='zeros'),
        tf.keras.layers.Dense(units=no_classes, activation='softmax', name='output_layer')
    ]

    BN_BEFORE_LAYERS = [
        tf.keras.layers.Flatten(input_shape=input_shape, name='input_layer'),
        tf.keras.layers.Dense(units=392, name='hidden1', kernel_initializer='glorot_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(units=196, name='hidden2', kernel_initializer='glorot_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(units=no_classes, activation='softmax', name='output_layer')
    ]

    BN_AFTER_LAYERS = [
        tf.keras.layers.Flatten(input_shape=input_shape, name='input_layer'),
        tf.keras.layers.Dense(units=392, activation='relu', name='hidden1', kernel_initializer='glorot_uniform',
                              bias_initializer='zeros'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=196, activation='relu', name='hidden2', kernel_initializer='glorot_uniform',
                              bias_initializer='zeros'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=no_classes, activation='softmax', name='output_layer')
    ]

    logging.info("Creating Model..")
    if stage == 'BASE_MODEL':
        LAYERS = BASE_LAYERS
    elif stage == 'KERNEL_INIT_MODEL':
        LAYERS = KERNEL_INIT_LAYERS
    elif stage == 'BN_BEFORE_MODEL':
        LAYERS = BN_BEFORE_LAYERS
    elif stage == 'BN_AFTER_MODEL':
        LAYERS = BN_AFTER_LAYERS

    model_ann = tf.keras.models.Sequential(LAYERS)

    logging.info("Compiling Model..")
    model_ann.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model_ann


def save_model(model_dir: str, model: Model, model_suffix: str) -> None:
    """
    args:
        model_dir: directory to save the model
        model: model object to save
        model_suffix: Suffix to save the model
    """
    create_directories([model_dir])
    model_file = os.path.join(model_dir, f"{model_suffix}.h5")
    model.save(model_file)
    logging.info(f"Saved model: {model_file}")


def save_history_plot(history, plot_dir: str, stage: str) -> None:
    """
    Args:
        history: History object for plotting loss/accuracy curves
        plot_dir: Directory to save plot files
        stage: Stage name for training
    """
    pd.DataFrame(history.history).plot(figsize=(10, 8))
    plt.grid(True)
    create_directories([plot_dir])
    plot_file = os.path.join(plot_dir, stage + "_loss_accuracy.png")
    plt.savefig(plot_file)
    logging.info(f"Loss accuracy plot saved: {plot_file}")


def get_callbacks(checkpoint_dir: str, tensorboard_logs: str, stage: str) -> list:
    """
    Args:
        checkpoint_dir: Directory to save the model at checkpoint
        tensorboard_logs: Directory to save tensorboard logs
        stage: Stage name for training
    Returns:
        callback_list: List of created callbacks
    """
    create_directories([checkpoint_dir, tensorboard_logs])
    tensorboard_cb = tf.keras.callbacks.TensorBoard(tensorboard_logs)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ckpt_file_path = os.path.join(checkpoint_dir, f"{stage}_ckpt_model.h5")
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_file_path, save_best_only=True)

    callback_list = [tensorboard_cb, early_stopping_cb, checkpoint_cb]
    logging.info(f"Callbacks created: {callback_list}")
    return callback_list
