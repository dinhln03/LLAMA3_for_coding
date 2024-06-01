#!/usr/bin/env py.test
#  -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import numpy as np
import pytest

import NN.layerversions.layers4 as layer


def test_fc():
    l1 = layer.FullyConnected(5, 10)
    x = np.ones((100, 5))
    y, c = l1.forward(x)
    assert y.shape == (100, 10)
    assert np.all(c == x)


def test_tanh():
    l = layer.Tanh()
    x = np.ones((100, 5))
    y, c = l.forward(x)
    assert y.shape == (100, 5)
    assert np.all(c == y)


@pytest.fixture()
def optim():
    return layer.sgd_optimiser(0.01)


def test_back_fc(optim):
    l1 = layer.FullyConnected(5, 10)

    x = np.ones((100, 5))
    dldy = np.random.randn(100, 10)

    dldx = l1.backward(dldy, x, optim)
    assert dldx.shape == (100, 5)


def test_back_tanh():
    l1 = layer.Tanh()
    x = np.random.randn(100, 5)
    dldy = np.random.randn(100, 5)

    dldx = l1.backward(dldy, np.tanh(x), optim)
    assert dldx.shape == (100, 5)


def test_network():
    from NN.loss import MSELoss
    x = np.random.randn(100, 10)
    y = np.random.randn(100, 3)
    net = layer.Network(
        layer.FullyConnected(10, 20),
        layer.Tanh(),
        layer.FullyConnected(20, 3),
        layer.Tanh()
    )

    mse = MSELoss()

    layer.train(net, (x, y), 10)

    yhat, _ = net.forward(x)
    initloss = mse.loss(y, yhat)
    layer.train(net, (x, y), 10)
    yhat, _ = net.forward(x)
    finloss = mse.loss(yhat, y)

    assert initloss > finloss
