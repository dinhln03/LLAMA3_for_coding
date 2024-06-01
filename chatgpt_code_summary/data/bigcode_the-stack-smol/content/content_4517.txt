import numpy as np
import math
from ml_from_scratch.activation_functions import Sigmoid
from ml_from_scratch.utils import make_diagonal


class LogisticRegression():
    """ Logistic Regression classifier.
    Parameters:
    -----------
    n_iters: int
        Number of iterations running gradient descent, default is 1000
    lr: float
        learning rate
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If
        false then we use Newton Method.
    """
    def __init__(self, n_iters=1000, lr=.1, gradient_descent=True):
        self.param = None
        self.n_iters = n_iters
        self.lr = lr
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y):
        self._initialize_parameters(X)
        # Tune parameters for n iterations
        for i in range(self.n_iters):
            # Make a new prediction
            y_pred = self.sigmoid(X.dot(self.param))
            if self.gradient_descent:
                # Move against the gradient of the loss function with
                # respect to the parameters to minimize the loss
                self.param -= self.lr * (y_pred - y).dot(X)
            else:
                # Make a diagonal matrix of the sigmoid gradient column vector
                diag_gradient = make_diagonal(self.sigmoid.gradient(X.dot(self.param)))
                # Batch opt:
                self.param = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).\
                    dot(X.T).dot(diag_gradient.dot(X).dot(self.param) + y - y_pred)

    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred

    def predict_proba(self, X):
        p_pred = self.sigmoid(X.dot(self.param))
        return p_pred
