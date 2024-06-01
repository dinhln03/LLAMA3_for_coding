import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def compile_model(network):
    """
    :param network dict: dictionary with network parameters
    :return: compiled model
    """

    
    model = lgb.LGBMRegressor(num_leaves=network.get('num_leaves', 31),
                             learning_rate=network.get('learning_rate', 0.1),
                             n_estimators=network.get('n_estimators', 20),
                             max_bin=network.get('max_bin', 1000),
                             colsample_bytree=network.get('colsample_bytree', 0.5),
                             subsample_for_bin=network.get('subsample_for_bin', 200000),
                             boosting_type=network.get('boosting_type', 'gbdt'),
                             num_iterations=network.get('num_iterations', 100),
                             extra_trees=network.get('extra_trees', False),
                             reg_sqrt= network.get('reg_sqrt', False),
                             bagging_freq = network.get('bagging_freq', 1),
                             bagging_fraction = network.get('bagging_fraction', 0.1))
    
    return model


def train_and_score(network, x_train, y_train, x_test, y_test):
    """

    :param network dict: dictionary with network parameters
    :param x_train array: numpy array with features for traning
    :param y_train array: numpy array with labels for traning
    :param x_test array: numpy array with labels for test
    :param y_test array: numpy array with labels for test
    :return float: score
    """

    model = compile_model(network)

    model.fit(x_train, y_train)

    y_pred = model.predict(np.array(x_test))
    

    true = y_test
    pred =  y_pred

    print(' R2 = ', r2_score(true, pred))
    
    return r2_score(true, pred), model
