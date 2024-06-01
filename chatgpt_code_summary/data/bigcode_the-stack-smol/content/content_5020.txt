import numpy as np


class Dense():

    def __init__(self, units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', input_shape=None):
        self._units = units
        self._activation = activation
        self._use_bias = use_bias
        self._kernal_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._bias = np.zeros((units, 1))

    def setPrevUnits(self, units):
        self._prevUnits = units
        self._weights = np.zeros((self._units, units))
        self._weights = np.random.standard_normal(
            size=self._weights.shape) * 0.01

    def forward(self, arr):
        out = self._weights.dot(arr) + self._bias
        if self._activation == "relu":
            out[out <= 0] = 0
        if self._activation == "softmax":
            out = self.softmax(out)
        return out

    def backwardFirst(self, dout, z):
        dw = dout.dot(z.T)
        db = np.sum(dout, axis=1)
        db = np.reshape(db, (db.shape[0], 1))
        return dw, db

    def backward(self, dout, next_weights, flat, z):
        dz = next_weights.T.dot(dout)
        if (self._activation == "relu"):
            dz[z <= 0] = 0
        dw = dz.dot(flat.T)
        db = np.sum(dz, axis=1).reshape(self._bias.shape)
        return dw, db, dz

    def softmax(self, X):
        out = np.exp(X)
        return out/np.sum(out)
