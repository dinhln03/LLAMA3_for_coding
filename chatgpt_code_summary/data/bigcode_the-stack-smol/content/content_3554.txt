from typing import Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from src.models.dnn_regressor_funcs import (
    _compile_model,
    _create_keras_model,
    _fit_model,
    _to_input_list,
)


def predict(model: tf.keras.Model, X_test: pd.DataFrame, cate_cols: list) -> np.array:
    """
    predict function
    Args:
        model: keras model fit by fit_model
        X_test: Test features
        cate_cols: categorical columns list

    Returns: y_pred

    """
    X_test_list = _to_input_list(df=X_test, cate_cols=cate_cols)
    y_pred = model.predict(X_test_list)
    return y_pred


def train(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.array],
    X_val: pd.DataFrame,
    y_val: Union[pd.Series, np.array],
    layers: list,
    num_classes: int,
    cate_cols: list,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    dropout_rate: float = 0.3,
) -> Tuple[tf.keras.callbacks.History, tf.keras.Model]:

    """
    Training main function that takes dataset and parameters as input and returns the trained model with history
    Args:
        X_train: Train features
        y_train: train labels
        X_val: Validation labels
        y_val: validation labels
        layers: List of nodes in hidden layers
        num_classes: Number of classes in target variable
        cate_cols: categorical columns list
        learning_rate: learning rate
        epochs: number of epochs
        batch_size: batch size
        dropout_rate: dropout rate

    Returns: history of training, trained model

    """

    X_train_list = _to_input_list(df=X_train, cate_cols=cate_cols)
    X_val_list = _to_input_list(df=X_val, cate_cols=cate_cols)

    # if len(y_train.shape) == 1:
    #     y_train_categorical = tf.keras.utils.to_categorical(
    #         y_train, num_classes=num_classes, dtype="float32"
    #     )
    #
    #     y_val_categorical = tf.keras.utils.to_categorical(
    #         y_val, num_classes=num_classes, dtype="float32"
    #     )
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    model = _create_keras_model(
        X_train=X_train,
        layers=layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        cate_cols=cate_cols,
    )

    _compile_model(model=model, num_classes=num_classes, learning_rate=learning_rate)
    history = _fit_model(
        model=model,
        X_train_list=X_train_list,
        y_train=y_train,
        X_val_list=X_val_list,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
    )

    return history, model
