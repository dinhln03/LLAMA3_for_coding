import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
dataset = load_boston()
X = dataset.data
y = dataset.target
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X-mean)/std
# print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
n_train = X_train.shape[0]
n_features = X_train.shape[1]

# 权重初始化
w = np.random.rand(n_features)
b = 1.1
lr = 0.001
epoches = 3000


def model(x):
    y_hat = w.dot(x)+b

    return y_hat


def loss_funtion(X, y):
    total_loss = 0
    n_samples = len(X)
    for i in range(n_samples):
        xi = X[i]
        yi = y[i]
        yi_hat = model(xi)
        total_loss += abs(yi_hat-yi)**2
    avg_loss = (1/n_samples)*total_loss
    return avg_loss


reg = 0.5
for epoch in range(epoches):
    sum_w = 0.0
    sum_b = 0.0
    for i in range(n_train):
        xi = X_train[i]
        yi = y_train[i]
        yi_hat = model(xi)
        sum_w += (yi_hat-yi)*xi
        sum_b += (yi_hat - yi)
    grad_w = (2/n_train)*sum_w+(2.0*reg*w)
    grad_b = (2/n_train)*sum_b  # 偏置项不做正则化处理
    w = w-lr*grad_w
    b = b-lr*grad_b

train_loss = loss_funtion(X_train, y_train)
test_loss = loss_funtion(X_test, y_test)
print(train_loss)
print(test_loss)
