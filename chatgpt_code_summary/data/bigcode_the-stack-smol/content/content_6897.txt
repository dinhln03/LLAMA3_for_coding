import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from torch.utils.data import DataLoader
from sklearn.ensemble import ExtraTreesClassifier
from parameters import *
from training.evaluation import Evaluate, ClassificationRanker
from training.feature_extraction import FeatureExtraction
from training.train_loop import train_loop
from training.utils import Utils, Datasets
import models as md

# Define Processor
print("1.\t" + str(device.type).capitalize() + " detected\n")

# Preprocess Data
utils = Utils()

featureExtraction = FeatureExtraction()
# validation data
print("2.\tProcessing Resume data for validation ...")
resume = utils.process_resumes(pth, categories, scores, query_name, feature_name)
featureExtraction.generate_features(resume, query_name, feature_name, resume_path)
# train data
print("3.\tProcessing Train data ...")
# utils.clean_save_data(data_train_path, data_test_path, data_valid_path, required_columns, clean_data_path)

# Load Data
print("4.\tLoading Data ...")
valid = utils.load_data(resume_path)
train_test = utils.load_data(clean_data_path)
output_dim = 1#len(train_test.y.unique())

# Train/Test Split
print("5.\tGetting Train/Test/Validation Data ...")
x_train, x_test, x_valid, y_train, y_test, y_valid, qid_train, qid_test, qid_valid = \
    utils.split_data(train_test, valid, .05)
print('6.\tTrain: {}\tTest: {}\tValid: {}\tOutput: {}'.format(x_train.shape, x_test.shape, x_valid.shape, output_dim))
print(
    '7.\tUnique Query Ids (train: {}\ttest: {}\tvalid: {})'.format(len(np.unique(qid_train)), len(np.unique(qid_test)),
                                                                   len(np.unique(qid_valid))))

# Define Model
# model = md.RNN(x_train.shape[1], output_dim, hidden2, 2)
# model = md.Model1(x_train.shape[1], hidden1, hidden2, hidden3, output_dim)
# model = md.Model2(output_dim)
model = md.Model4(x_train.shape[1], output_dim)
model.to(device)
print("8.\tModel defined and moved to " + str(device.__str__()))

# Parameters
optimizer = Optimizer(model.parameters())
scheduler = scheduler(optimizer)
print("9.\tCriterion set as " + str(criterion.__str__()))
print("10.\tOptimizer set as " + str(optimizer.__str__()))

# Data Loader
train_dataset = Datasets(y_train, x_train, qid_train)
test_dataset = Datasets(y_test, x_test, qid_test)
valid_dataset = Datasets(y_valid, x_valid, qid_valid)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=56, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
train_qid, train_labels, train_features = next(iter(train_loader))
print("11.\tDataLoader Shapes-> QID: {}\tLabel: {}\tFeatures: {}".format(train_qid.size(), train_labels.size(),
                                                                         train_features.size()))

# NN Model
print("12.\tTrain loop")
# train_loop(model, epochs, optimizer, criterion, train_loader, test_loader, valid_loader, k_rank,
#            printing_gap, saved_model_device, model_path, device, PIK_plot_data, scheduler)

# Regressor Model
# rfr = RandomForestRegressor(n_estimators=200, min_samples_split=5, random_state=1, n_jobs=-1)
# rfr.fit(x_train, y_train)
# Evaluate().print_evaluation(rfr, x_train, y_train, qid_train, k_rank)
# Evaluate().print_evaluation(rfr, x_test, y_test, qid_test, k_rank)
# Evaluate().print_evaluation(rfr, x_valid, y_valid, qid_valid, k_rank)
# Evaluate().save_model(rfr, reg_model_path)

# SVM Model
sm = svm.SVR()
sm.fit(x_train, y_train)
Evaluate().print_evaluation(sm, x_train, y_train, qid_train, k_rank)
Evaluate().print_evaluation(sm, x_test, y_test, qid_test, k_rank)
Evaluate().print_evaluation(sm, x_valid, y_valid, qid_valid, k_rank)
Evaluate().save_model(sm, svm_model_path)


# Classifier Model
# etc = ClassificationRanker(LogisticRegression(C=1000))
# etc.fit(x_train, y_train)
# Evaluate().print_evaluation(etc, x_train, y_train, qid_train, k_rank)
# Evaluate().print_evaluation(etc, x_test, y_test, qid_test, k_rank)
# Evaluate().print_evaluation(etc, x_valid, y_valid, qid_valid, k_rank)
#
# yp = rfr.predict(x_valid)
# for i, j, k in zip(qid_valid, y_valid, yp):
#     print(i, j, k)
