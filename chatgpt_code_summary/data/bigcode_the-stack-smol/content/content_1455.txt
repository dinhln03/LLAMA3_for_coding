
"""
Purpose: Unsupervised learning sampler
Date created: 2020-11-06

Ref repo: https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries

Local folder: C:/Users/Work1/Desktop/Info/GitHub/python-examples-main/notebook-samples/unsupervised

Contributor(s):
    Mark M.
"""

import os
from pathlib import Path

# Set local folder if developing/debugging
myuser = os.environ["username"]
PROJECT_FOLDER = Path(rf"C:\Users\{myuser}\Desktop\Info\GitHub\python-examples-main\notebook-samples\unsupervised")
os.chdir(PROJECT_FOLDER)


from UnsupervisedTSRepo import scikit_wrappers

import gc
import zipfile
import requests
from io import BytesIO, StringIO

# Data sci and dat processing imports
import scipy as sp
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import sklearn
from sklearn import cluster
from sklearn import neighbors

import torch
import torch.nn as nn
import torch.optim as optim


pd.set_option("mode.chained_assignment", None)
pd.set_option("display.width", 120)
pd.set_option("display.date_yearfirst", True)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_info_rows", 10000)

gc.enable()

# Check for CUDA
CUDA_TF: bool = False
if torch.cuda.is_available():
    print("Using CUDA...")
    CUDA_TF = True

GPU = 0



zip_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"

def import_zipfile_data(URL = zip_url):
    with requests.Session() as s:
        tmp = s.get(URL)
        with zipfile.ZipFile(BytesIO(tmp.content)) as zfo:
            with zfo.open("household_power_consumption.txt") as zfi:
                tmp = StringIO(zfi.read().decode("utf-8"))
                data_ = pd.read_csv(tmp, sep=";", decimal=",", header=0, low_memory=False)
                del tmp
    return data_


data = import_zipfile_data(zip_url)

data.loc[:, "Date"] = pd.to_datetime(data.loc[:, "Date"], yearfirst=True)
data.loc[:, "Time"] = pd.to_datetime(data.loc[:, "Time"], format="%H:%M:%S").dt.time


#dataset = data.transpose(pd.array(data))[2].reshape(1, 1, -1)


# Update missing values with the "last seen" value.
# This probably works better for timeseries than other data
# since order is important here.
dataset = np.transpose(np.array(data))[2].reshape(1, 1, -1)
for idx in range(np.shape(dataset)[2]):
    if dataset[0, 0, idx] == "?":
        dataset[0, 0, idx] = dataset[0, 0, idx - 1]
dataset = dataset.astype(np.float32)


# Create training and testing sets
train = dataset[:, :, :500000]
test = dataset[:, :, 500000:]

# Normalization

mu_ = np.mean(dataset)
sigma_ = np.std(dataset)

normalize = lambda d, mean, sigma: (d - mean) / sigma

dataset = normalize(dataset, mu_, sigma_)
train = normalize(train, mu_, sigma_)
test = normalize(test, mu_, sigma_)

print(f"Normalized data set metrics:\n\tMean: {np.mean(dataset)}\n\tVariance: {np.var(dataset)}")

# Feature learning

# Train new model?
training = True

model_path =  PROJECT_FOLDER.joinpath(r"data\HouseholdPowerConsumption_yearly")

# hyperparams = {
#     "batch_size": 1,
#     "channels": 30,
#     "compared_length": None,
#     "depth": 10,
#     "nb_steps": 400,
#     "in_channels": 1,
#     "kernel_size": 3,
#     "penalty": None,
#     "early_stopping": None,
#     "lr": 0.001,
#     "nb_random_samples": 10,
#     "negative_penalty": 1,
#     "out_channels": 160,
#     "reduced_size": 80,
#     "cuda": CUDA_TF,
#     "gpu": GPU
# }

# encoder_yearly = scikit_wrappers.CausalCNNEncoderClassifier()
# encoder_yearly.set_params(**hyperparams)


# if training:
#     encoder_yearly.fit_encoder(train, save_memory=True, verbose=True)
#     encoder_yearly.save_encoder(model_path.as_posix())
# else:
#     encoder_yearly.load_encoder(model_path.as_posix())



torch.cuda.empty_cache()













"""" For local zipfile data
from io import StringIO
with zipfile.ZipFile("household_power_consumption.zip") as zfo:
    with zfo.open("household_power_consumption.txt") as zfi:
        tmp = StringIO(zfi.read().decode("utf-8"))
        data = pd.read_csv(tmp, sep=";", decimal=",", header=0, low_memory=False)
        del tmp
"""

"""
import hmac
import pickle
import hashlib
import binascii
def create_sha256_signature(key, message):
    byte_key = binascii.unhexlify(key)
    message = message.encode()
    return hmac.new(byte_key, message, hashlib.sha256).hexdigest().upper()


create_sha256_signature("E49756B4C8FAB4E48222A3E7F3B97CC3", "TEST STRING")
"""
