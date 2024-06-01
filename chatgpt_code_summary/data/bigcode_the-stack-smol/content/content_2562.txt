#Neural Networks
#MLP classifier is optimal algorithm for classifications

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train_clean, y_train)

clf.predict(X_test_clean)
scoreN = clf.score(X_test_clean, y_test)
print(scoreN)