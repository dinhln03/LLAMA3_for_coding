#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """

import h5py
import torch
import torch.utils
import torch.utils.data

from .h5_mnist_data import download_binary_mnist


def load_binary_mnist(cfg, **kwcfg):
    fname = cfg.data_dir / "binary_mnist.h5"
    if not fname.exists():
        print("Downloading binary MNIST data...")
        download_binary_mnist(fname)
    f = h5py.File(str(fname), "r")
    x_train = f["train"][::]
    x_val = f["valid"][::]
    x_test = f["test"][::]
    train = torch.utils.data.TensorDataset(torch.from_numpy(x_train))
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=cfg.batch_size, shuffle=True, **kwcfg
    )
    validation = torch.utils.data.TensorDataset(torch.from_numpy(x_val))
    val_loader = torch.utils.data.DataLoader(
        validation, batch_size=cfg.test_batch_size, shuffle=False
    )
    test = torch.utils.data.TensorDataset(torch.from_numpy(x_test))
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=cfg.test_batch_size, shuffle=False
    )
    return train_loader, val_loader, test_loader
