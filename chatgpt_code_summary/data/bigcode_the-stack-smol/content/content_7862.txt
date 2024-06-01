# Lab 2 Linear Regression
import tensorflow as tf

tf.set_random_seed(777)  # seed 설정

# training data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# regerssion 결과는 W = 1, b = 0 이라는 것을 알 수 있음
# but tensorflow로 training 시켜서 해보기!!
# W와 b는 어떻게 달라질까?

# tf.Variable() : tensorflow가 사용하는 변수(trainable variable)
# tf.random_normal([1]) : normal dist에서 1개의 난수 생성
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Linear regression model
hypothesis = x_train * W + b

# cost/loss function (MSE)
# tf.square() : 제곱해주는 tf 함수
# tf.reduce_mean() : mean 구해주는 tf 함수
# hypothesis(y_hat), y_train(true value)
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# GradientDescent
# Minimize
# learning rate=0.01로 training 시킴 => gradient descent로 인해 조금씩 true에 가까워짐
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# session 객체 생성(tf graph 객체 생성)
sess = tf.Session()
# 모든 tf variavle 초기화
sess.run(tf.global_variables_initializer())

# Fit
# 2001번 최적화 시킴!!!
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:  # 다 뽑으면 너무 많으니까 몇개만 뽑기 위해서
        # step(몇 번째인지?), cost(mse), W(weight), b(bias)
        print(step, sess.run(cost), sess.run(W), sess.run(b))

# Learns best fit W:[ 1.],  b:[ 0.]

'''
0 2.82329 [ 2.12867713] [-0.85235667]
20 0.190351 [ 1.53392804] [-1.05059612]
40 0.151357 [ 1.45725465] [-1.02391243]
...
1920 1.77484e-05 [ 1.00489295] [-0.01112291]
1940 1.61197e-05 [ 1.00466311] [-0.01060018]
1960 1.46397e-05 [ 1.004444] [-0.01010205]
1980 1.32962e-05 [ 1.00423515] [-0.00962736]
2000 1.20761e-05 [ 1.00403607] [-0.00917497]
'''