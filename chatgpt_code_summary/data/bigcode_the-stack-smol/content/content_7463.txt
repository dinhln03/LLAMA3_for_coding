#  -*- coding: utf-8 -*-
"""
Key classification
using multiclass Support Vector Machine (SVM)
reference: 

Date: Jun 05, 2017
@author: Thuong Tran
@Library: scikit-learn
"""


import os, glob, random
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.cross_validation import KFold
import time
import codecs
import matplotlib.pyplot as plt
import itertools


NEW_LINE = '\r\n'
TRAIN_SIZE = 0.8


def build_data_frame(data_dir):  
  # folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
  dirs = next(os.walk(data_dir))[1] # [0]: path, [1]: folders list, [2] files list
  
  class_names = []
  total_amount = []
  train_amount = []
  test_amount = []

  train_data = DataFrame({'value': [], 'class': []})
  test_data = DataFrame({'value': [], 'class': []})

  for d in dirs:
    tmp_dir = os.path.join(data_dir, d)
    rows = []
    index = []
    for f in glob.glob(os.path.join(tmp_dir, '*.txt')):
      with open(f, encoding="latin1") as fc:
        value = [line.replace('\n', '').replace('\r', '').replace('\t', '') 
                    for line in fc.readlines()]
        value = '. '.join(value)
        rows.append({'value': value, 'class': d})
        index.append(f)

    tmp_df = DataFrame(rows, index=index)
    size = int(len(tmp_df) * TRAIN_SIZE)
    train_df, test_df = tmp_df.iloc[:size], tmp_df.iloc[size:]

    train_data = train_data.append(train_df)
    test_data = test_data.append(test_df)

    class_names.append(d)

    total_amount.append(len(os.listdir(tmp_dir)))
    train_amount.append(len(train_df))
    test_amount.append(len(test_df))
  
  tmp_arr = np.array([total_amount, train_amount, test_amount])
  print (DataFrame(tmp_arr, ['Total', 'Train', 'Test'], class_names))
  
  train_data = train_data.reindex(np.random.permutation(train_data.index))
  test_data = test_data.reindex(np.random.permutation(test_data.index))
  return train_data, test_data, class_names


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
    else:
      print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i, j],
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
  data_dir = '/Users/thuong/Documents/tmp_datasets/SI/TrainValue'
  train_data_df, test_data_df, class_names = build_data_frame(data_dir)

  pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf_transformer',  TfidfTransformer()),
    ('classifier', LinearSVC())])

  ######### One-KFolds ##############################
  train_data, test_data = train_data_df['value'].values, test_data_df['value'].values
  train_target, test_target = train_data_df['class'].values, test_data_df['class'].values
  
  pipeline.fit(train_data, train_target)
  predictions = pipeline.predict(test_data)

  cnf_matrix = confusion_matrix(test_target, predictions)
  print('Confusion matrix with one-fold: ')
  print(cnf_matrix)
  print("Score with one-fold: %s" % precision_score(test_target, predictions, average = 'weighted'))
  print("Score with one-fold: %s" % precision_score(test_target, predictions, average = None))

  # ######### KFolds ##############################
  # k_fold = KFold(n=len(data_frame), n_folds=6)
  # scores = []
  # confusion = np.array([[0, 0], [0, 0]])
  # for train_indices, test_indices in k_fold:
  #   train_text = data_frame.iloc[train_indices]['text'].values
  #   train_label = data_frame.iloc[train_indices]['class'].values
  #   test_text = data_frame.iloc[test_indices]['text'].values
  #   test_label = data_frame.iloc[test_indices]['class'].values

  #   pipeline.fit(train_text, train_label)
  #   predictions = pipeline.predict(test_text)

  #   confusion += confusion_matrix(test_label, predictions)
  #   score = f1_score(test_label, predictions, pos_label = SPAM)
  #   scores.append(score)

  # print('Confusion matrix with 6-fold: ')
  # print(confusion)
  # print('Score with 6-fold: %s' % (sum(scores)/len(scores)))  

  # Plot non-normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=class_names,
                        title='Confusion matrix, without normalization')
  # Plot normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                        title='Normalized confusion matrix')
  plt.show()


if __name__ == "__main__":
  main()