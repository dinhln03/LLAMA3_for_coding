import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from .metrics import mse_score, rmse_score, r2_score, mae_score
from ..features.build_features import StandardScaler, MinMaxScaler

class LinearRegressor():
    """Linear regressor"""

    def __init__(self, method='normal_equation', normalize=False, lr=0.01, epochs=1000, add_intercept=False):
        assert method in ['normal_equation', 'gradient_descent'], "Method not supported. Supported methods are 'normal_equation' and 'gradient_descent'"
        self.method = method
        self.normalize = normalize
        self.add_intercept = add_intercept
        self._weights = None

        if self.method == 'gradient_descent':
            self.lr = lr
            self.epochs = epochs

        if self.normalize:
            self._feature_scaler = MinMaxScaler()
            self._target_scaler = MinMaxScaler()
    
    def fit(self, X, y):
        """Fit the model to the data"""
        
        if self.normalize:
            X = self._feature_scaler.fit_transform(X)
            y = self._target_scaler.fit_transform(y)

        X = X.to_numpy()
        if self.add_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        y = y.to_numpy()

        if self.method == 'normal_equation':
            self._weights = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

        else:
            # mse_new = np.inf
            self._weights = np.zeros(X.shape[1])
            self.cost_history = [0] * self.epochs

            for i in range(self.epochs):
                grad = np.dot(X.T, np.dot(X, self._weights) - y) / y.shape[0]
                self._weights = self._weights - self.lr * grad
                self.cost_history[i] = mse_score(y, np.dot(X, self._weights))

            #     if (rmse_new > rmse_old):
            #         print("Stopped at iteration {}".format(i))
            #         break
            plt.scatter(range(self.epochs), self.cost_history)
            plt.xlabel('epoch')
            plt.ylabel('mse')

    def predict(self, X):
        """Use the fitted model to predict on data"""

        assert self._weights is not None, "Model needs to be fitted first. Use the fit method"

        if self.normalize:
            X = self._feature_scaler.transform(X)

        X = X.to_numpy()        
        if self.add_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
            
        y_pred = np.dot(X, self._weights)
        if self.normalize:
            y_pred = self._target_scaler.inverse_transform(y_pred)

        return np.round(y_pred, 2)
    
    def get_weights(self):
        """Get weights from the fitted model"""

        assert self._weights is not None, "Model needs to be fitted first. Use the fit method"
        return self._weights

    def score(self, X, y, metric='r2'):
        """Score the model"""

        assert metric in ['r2', 'rmse', 'mae'], "Metric not supported. Supported metrics are 'r2', 'rmse' and 'mae'"

        y_pred = self.predict(X)    
        
        if metric == 'r2':
            score = r2_score(y, y_pred)
        elif metric == 'rmse':
            score = rmse_score(y, y_pred)
        elif metric == 'mae':
            score = mae_score(y, y_pred)

        return score