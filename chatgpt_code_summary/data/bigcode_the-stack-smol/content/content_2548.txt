## -------------------------------------------------------- ##
#   Trab 2 IA 2019-2
#
#   Rafael Belmock Pedruzzi
#
#   probOneR.py: implementation of the probabilistic OneR classifier.
#
#   Python version: 3.7.4
## -------------------------------------------------------- ##

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import confusion_matrix
from itertools import product, zip_longest, accumulate
from random import random

class Prob_OneR(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        # check that x and y have correct shape
        X, y = check_X_y(X,y)
        # store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.y_ = y

        kbd = KBinsDiscretizer(n_bins = len(np.unique(y)), encode='ordinal')
        X = kbd.fit_transform(X)
        self.X_ = X
        self.kbd_ = kbd

        cm_list = []
        hits = []
        for i in X.T:
            cm = contingency_matrix(i, y)
            cm_list.append(cm)
            hits.append(sum(max(k) for k in cm))

        rule = np.argmax(hits) # chosen rule
        self.r_ = rule

        rule_cm = cm_list[rule]
        class_selector = []
        for i, c in enumerate(rule_cm):
            cSum = sum(c)
            probRatio = [ (i/cSum) for i in c]
            # Building the "partitions" of the roulette:
            probRatio = list(accumulate(probRatio))
            class_selector.append(probRatio)
        self.class_selector = class_selector

        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        X = self.kbd_.transform(X)

        y = []
        for i in X[:,self.r_]:
            probRatio = self.class_selector[int(i)]
            # Selecting a random element:
            selector = random()
            for i in range(len(probRatio)):
                if selector <= probRatio[i]:
                    y.append(self.classes_[i])
                    break
        return y


# from sklearn import datasets
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import f1_score

# nn= Prob_OneR()
# iris = datasets.load_iris()
# x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size = 0.4, random_state = 0)
# nn.fit(x_train, y_train)
# y_pred = nn.predict(x_test)
# print(y_test)
# print(y_pred)
# score = cross_val_score(nn, x_train, y_train, cv = 5)
# print(score)
