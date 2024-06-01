#!/usr/bin/env python
import numpy as np
import scipy
from shared_utils import ArrayUtils


class SkellamMetrics:
    def __init__(self, x_metrics, y_metrics, y_hat, model, l0, l1, training_values):
        self._y = y_metrics
        self._y_hat = y_hat
        self.model = model
        self.l0 = ArrayUtils.convert_to_array(l0)
        self.l1 = ArrayUtils.convert_to_array(l1)
        self.training_values = training_values
        self._x0, self._x1 = self.split_or_duplicate_x(x_metrics)
        self.max_ll = self.model.fun
        self.coeff_size = self._x0.shape[1]
        self.lambda_0_coefficients = self.model.x[0 : self.coeff_size].reshape(-1, 1)
        self.lambda_1_coefficients = self.model.x[self.coeff_size :].reshape(-1, 1)
        self.train_length = len(training_values[0])

    @staticmethod
    def split_or_duplicate_x(x):
        return ArrayUtils.split_or_duplicate_x(x, False)

    def sse(self):
        return ((self._y - self._y_hat) ** 2).sum()

    def _y_bar(self):
        return self._y.mean()

    def sst(self):
        return ((self._y - self._y_bar()) ** 2).sum()

    def r2(self):
        """Calculate R2 for either the train model or the test model"""
        sse_sst = self.sse() / self.sst()
        return 1 - sse_sst

    def adjusted_r2(self):
        """Calculate adjusted R2 for either the train model or the test model"""
        r2 = self.r2()
        return 1 - (1-r2)*(self.train_length - 1)/(self.train_length - self.coeff_size - 1)

    def log_likelihood(self):
        """Returns the maximum of the log likelihood function"""
        return self.max_ll

    def aic(self):
        return 2*self.coeff_size - 2*np.log(self.max_ll)

    def bic(self):
        return self.coeff_size*np.log(self.train_length) - 2*np.log(self.max_ll)

    def _calculate_lambda(self):
        """Create arrays for our predictions of the two Poisson distributions
        """
        _lambda0 = ArrayUtils.convert_to_array(
            np.exp(np.squeeze(self._x0 @ self.lambda_0_coefficients))
        )
        _lambda1 = ArrayUtils.convert_to_array(
            np.exp(np.squeeze(self._x1 @ self.lambda_1_coefficients))
        )
        return _lambda0, _lambda1

    def _calculate_v(self):
        """Create diagonal matrix consisting of our predictions of the Poisson distributions
        """
        _lambda0, _lambda1 = self._calculate_lambda()
        _v0 = np.diagflat(_lambda0)
        _v1 = np.diagflat(_lambda1)
        return _v0, _v1

    def _calculate_w(self):
        """Create a diagonal matrix consisting of the difference between our predictions of the 2 Poisson distributions
        with their observed values
        """
        _lambda0, _lambda1 = self._calculate_lambda()
        _w0 = np.diagflat((self.l0 - _lambda0.reshape(-1, 1)) ** 2)
        _w1 = np.diagflat((self.l1 - _lambda1.reshape(-1, 1)) ** 2)
        return _w0, _w1

    def _calculate_robust_covariance(self):
        """Calculate robust variance covariance matrices for our two sets of coefficients
        """
        _v0, _v1 = self._calculate_v()
        _w0, _w1 = self._calculate_w()
        _robust_cov0 = (
            np.linalg.inv(np.dot(np.dot(self._x0.T, _v0), self._x0))
            * np.dot(np.dot(self._x0.T, _w0), self._x0)
            * np.linalg.inv(np.dot(np.dot(self._x0.T, _v0), self._x0))
        )
        _robust_cov1 = (
            np.linalg.inv(np.dot(np.dot(self._x1.T, _v1), self._x1))
            * np.dot(np.dot(self._x1.T, _w1), self._x1)
            * np.linalg.inv(np.dot(np.dot(self._x1.T, _v1), self._x1))
        )
        return _robust_cov0, _robust_cov1

    def _calculate_robust_standard_errors(self):
        """Calculate robust standard errors for our two sets of coefficients by taking the square root of the diagonal
        values in the variance covariance matrices
        """
        _robust_cov0, _robust_cov1 = self._calculate_robust_covariance()
        _std_error0 = np.sqrt(np.diag(_robust_cov0))
        _std_error1 = np.sqrt(np.diag(_robust_cov1))
        return _std_error0, _std_error1

    def _calculate_z_values(self):
        """Calculate z statistics for our two sets of coefficients
        """
        _std_error0, _std_error1 = self._calculate_robust_standard_errors()
        _z_values0 = self.lambda_0_coefficients[:, 0] / _std_error0
        _z_values1 = self.lambda_1_coefficients[:, 0] / _std_error1
        return _z_values0, _z_values1

    def _calculate_p_values(self):
        """Calculate p values for our two sets of coefficients
        """
        _z_values0, _z_values1 = self._calculate_z_values()
        _p_values0 = scipy.stats.norm.sf(abs(_z_values0)) * 2
        _p_values1 = scipy.stats.norm.sf(abs(_z_values1)) * 2
        return _p_values0, _p_values1
