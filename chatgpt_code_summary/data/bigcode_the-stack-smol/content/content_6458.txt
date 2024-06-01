import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVC

from O3 import prepare_data

from utils import plot_decision_regions

X_train, X_test, y_train, y_test = prepare_data(standardize=True,
                                                split=True)

svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)

svm.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined, classifier=svm,
                      test_idx=range(105, 150))

plt.xlabel('petal length [standardize]')
plt.ylabel('petal width [standardize]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

print(f'Accuracy: {svm.score(X_test, y_test) * 100}%')
