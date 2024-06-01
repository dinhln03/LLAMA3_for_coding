# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 16:02:11 2019

@author: Christian Zwinkels-Valero
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(z))

def d_sigmoid(z):
    return sigmoid(z)*(1 - sigmoid(z))

def relu(z, d=False):
    if d == False:
        f = np.maximum(0.001*z, z)
    else:
        f = np.where(z >= 0, 1, 0.001)
    return f


# Data processing
data = pd.read_csv("IRISS.csv", header=None, skiprows=1)
data = data.sample(frac=1)
X = data[data.columns[0:data.shape[-1] - 1]].to_numpy()
X = (X - np.mean(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
X = X.T
y = np.array([data[data.shape[-1] - 1].to_numpy()])

# Initialization
layer_sizes = (X.shape[0], 12, 8, 4, y.shape[0])
weight_sizes = [(y, x) for y, x in zip(layer_sizes[1:], layer_sizes[0:])]
weights = [np.random.rand(l[0], l[1])*np.sqrt(1/l[1]) for l in weight_sizes]
biases = [np.zeros((i, 1)) for i in layer_sizes[1:]]

# Foward propagation
def feedforward(data_in, Ws, Bs):
    Z = []
    A = [data_in]  # First activation layer is the inputs

    # Hidden layer computation
    for i in range(len(Ws) - 1):
        z = np.dot(Ws[i], A[-1]) + Bs[i]
        a = relu(z, d=False)
        Z.append(z)
        A.append(a)

    # Ouput layer computation
    z = np.dot(Ws[-1], A[-1]) + Bs[-1]
    Z.append(z)
    a = sigmoid(z)
    A.append(a)
    return Z, A


# Calculating the costs
def costs(data_in, outputs, Ws, Bs):
    Z, pred = feedforward(data_in, Ws, Bs)
    delta = []
    dj_dw = []

    # Loss computation
    loss = -1*(outputs*np.log(pred[-1]) + (1-outputs)*np.log(1 - pred[-1]))
    loss = np.mean(loss)

    # Final layer derivatives
    dj_da = -1*((outputs[-1]/pred[-1]) + (1 - outputs)/(1 - pred[-1]))
    da_dz = d_sigmoid(Z[-1])
    delta.append(dj_da*da_dz)

    # Deltas calculation
    for i in range(1, len(Ws)):
        d = np.dot(Ws[-i].T, delta[-i]) * relu(Z[-i - 1], d=True)
        delta.insert(0, np.mean(d, axis=1, keepdims=True))
    delta[-1] = np.mean(delta[-1])

    # dj_dw calculation
    for i in range(1, len(pred)):
        b = []
        A = pred[-i - 1].T
        for a in A:
            b.append(np.dot(delta[-i], [a]))
        d = np.zeros(weight_sizes[-i])
        for s in b:
            d += s
        d /= len(d)
        dj_dw.insert(0, d)
    return loss, delta, dj_dw

def train(data_in, outputs, Ws, Bs, iters, alpha):
    c_his = []
    i_his = []
    for i in range(iters):
        c, Bu, Wu = costs(data_in, outputs, Ws, Bs)
        for j in range(len(Ws)):
            Bs[j] = np.add(Bs[j], np.multiply(-alpha, Bu[j]))
            Ws[j] = np.add(Ws[j], np.multiply(-alpha, Wu[j]))
        c_his.append(c)
        i_his.append(i)
    plt.plot(i_his, c_his)
    plt.show()
