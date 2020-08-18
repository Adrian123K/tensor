import tensorflow as tf
import numpy as np
import warnings
import os
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# 136
#
# # 은닉1층
# x = tf.placeholder('float',[None,784])
# W1 = tf.get_variable(name='W1', shape=[784,50], initializer = tf.contrib.layers.xavier_initializer())
# b1 = tf.Variable(tf.ones([1,50]))
#
# y = tf.matmul(x, W1) + b1
# y_hat = tf.nn.relu(y)
#
# # 출력층
# W2 = tf.Variable(tf.random_uniform([50,10],-1,1))
# b2 = tf.Variable(tf.ones([1,10]))
#
# z = tf.matmul(y_hat,W2) + b2
# z_hat = tf.nn.softmax(z)
# y_predict = tf.argmax(z_hat, axis=1)
#
# # 정확도 확인
# y_onehot = tf.placeholder('float',[None,10]) # 정답 데이터를 담을 배열
# y_label = tf.argmax(y_onehot, axis=1)
#
# correction_prediction = tf.equal(y_predict, y_label)
# accuracy = tf.reduce_mean(tf.cast(correction_prediction,'float'))
#
# # 오차 확인
# loss = -tf.reduce_sum(y_onehot * tf.log(z_hat+0.0000001), axis=1)
# rs = tf.reduce_mean(loss)
#
# # 학습
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)
#
# # 변수 초기화
# init = tf.global_variables_initializer()
#
# # 그래프 실행
# sess = tf.Session()
# sess.run(init)
#
# for i in range(1,601*20):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     batch_x_test, batch_y_test = mnist.test.next_batch(100)
#     sess.run(train, feed_dict={x:batch_xs, y_onehot:batch_ys})
#     if not i % 600:
#         print(i//600,"에폭 훈련데이터 정확도 : ",sess.run(accuracy, feed_dict={x:batch_xs, y_onehot:batch_ys}),"\t","테스트 데이터 정확도:", sess.run(accuracy, feed_dict={x:batch_x_test, y_onehot:batch_y_test}))

# 137

tf.reset_default_graph() # 텐서 그래프 초기화 하는 코드

# # 은닉1층
# x = tf.placeholder('float',[None,784])
# W1 = tf.get_variable(name="W1", shape=[784,50], initializer = tf.contrib.layers.variance_scaling_initializer())
# b1 = tf.Variable(tf.ones([1,50]))
#
# y = tf.matmul(x, W1) + b1
# y_hat = tf.nn.relu(y)
#
# # 출력층
# W2 = tf.get_variable(name="W2", shape=[50,10], initializer = tf.contrib.layers.variance_scaling_initializer())
# b2 = tf.Variable(tf.ones([1,10]))
#
# z = tf.matmul(y_hat,W2) + b2
# z_hat = tf.nn.softmax(z)
# y_predict = tf.argmax(z_hat, axis=1)
#
# # 정확도 확인
# y_onehot = tf.placeholder('float',[None,10]) # 정답 데이터를 담을 배열
# y_label = tf.argmax(y_onehot, axis=1)
#
# correction_prediction = tf.equal(y_predict, y_label)
# accuracy = tf.reduce_mean(tf.cast(correction_prediction,'float'))
#
# # 오차 확인
# loss = -tf.reduce_sum(y_onehot * tf.log(z_hat+0.0000001), axis=1)
# rs = tf.reduce_mean(loss)
#
# # 학습
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)
#
# # 변수 초기화
# init = tf.global_variables_initializer()
#
# # 그래프 실행
# sess = tf.Session()
# sess.run(init)
#
# for i in range(1,601*20):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     batch_x_test, batch_y_test = mnist.test.next_batch(100)
#     sess.run(train, feed_dict={x:batch_xs, y_onehot:batch_ys})
#     if not i % 600:
#         print(i//600,"에폭 훈련데이터 정확도 : ",sess.run(accuracy, feed_dict={x:batch_xs, y_onehot:batch_ys}),"\t","테스트 데이터 정확도:", sess.run(accuracy, feed_dict={x:batch_x_test, y_onehot:batch_y_test}))

# 138
# # 은닉1층
# x = tf.placeholder('float',[None,784])
# W1 = tf.get_variable(name="W1", shape=[784,100], initializer = tf.contrib.layers.variance_scaling_initializer())
# b1 = tf.Variable(tf.ones([1,100]))
#
# y = tf.matmul(x, W1) + b1
# y_hat = tf.nn.relu(y)
#
# # 은닉2층
# W2 = tf.get_variable(name="W2", shape=[100,50], initializer = tf.contrib.layers.variance_scaling_initializer())
# b2 = tf.Variable(tf.ones([1,50]))
#
# y2 = tf.matmul(y_hat,W2) + b2
# y2_hat = tf.nn.relu(y2)
#
# # 출력층
# W3 = tf.get_variable(name="W3", shape=[50,10], initializer = tf.contrib.layers.variance_scaling_initializer())
# b3 = tf.Variable(tf.ones([1,10]))
#
# z = tf.matmul(y2_hat,W3) + b3
# z_hat = tf.nn.softmax(z)
# y_predict = tf.argmax(z_hat, axis=1)
#
# # 정확도 확인
# y_onehot = tf.placeholder('float',[None,10]) # 정답 데이터를 담을 배열
# y_label = tf.argmax(y_onehot, axis=1)
#
# correction_prediction = tf.equal(y_predict, y_label)
# accuracy = tf.reduce_mean(tf.cast(correction_prediction,'float'))
#
# # 오차 확인
# loss = -tf.reduce_sum(y_onehot * tf.log(z_hat+0.0000001), axis=1)
# rs = tf.reduce_mean(loss)
#
# # 학습
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)
#
# # 변수 초기화
# init = tf.global_variables_initializer()
#
# # 그래프 실행
# sess = tf.Session()
# sess.run(init)
#
# for i in range(1,601*20):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     batch_x_test, batch_y_test = mnist.test.next_batch(100)
#     sess.run(train, feed_dict={x:batch_xs, y_onehot:batch_ys})
#     if not i % 600:
#         print(i//600,"에폭 훈련데이터 정확도 : ",sess.run(accuracy, feed_dict={x:batch_xs, y_onehot:batch_ys}),"\t","테스트 데이터 정확도:", sess.run(accuracy, feed_dict={x:batch_x_test, y_onehot:batch_y_test}))

###################################################### 텐서 플로우 배치 정규화 및 Convolution

import tensorflow as tf
import numpy as np
import warnings
import os
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

tf.reset_default_graph() # 텐서 그래프 초기화 하는 코드

# 입력층
x = tf.placeholder('float',[None,784])
x1 = tf.reshape(x,[-1,28,28,1]) # 흑백사진, 1층, batch 개수를 모르므로 -1. 2차원 -> 4차원으로 변경

# Convolution 1층
W1 = tf.Variable(tf.random_normal([5,5,1,32], stdde=0.01)) # 필터 32개 생성
b1 = tf.Variable(tf.ones([32])) # 숫자 1로 채워진 bias 생성
y1 = tf.nn.conv2d(x1, W1, strides=[1,1,1,1], padding='SAME')
y1 = y1 + b1
y1 = tf.nn.relu(y1)
y1 = tf.nn.max_pool(y1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # ksize : 필터 사이즈
y1 = tf.reshape(y1, [-1,14*14*32]) # y1 4차원 -> 2차원

# 완전연결계층 1층
W2 = tf.get_variable(name="W2", shape=[14*14*32,100], initializer = tf.contrib.layers.variance_scaling_initializer())
b2 = tf.Variable(tf.ones([1,100]))

y2 = tf.matmul(y1, W2) + b2

batch_y2 = tf.contrib.layers.batch_norm(y2, True)

y2_hat = tf.nn.relu(batch_y2)

# 완전연결계층 2층
W3 = tf.get_variable(name="W3", shape=[100,50], initializer = tf.contrib.layers.variance_scaling_initializer())
b3 = tf.Variable(tf.ones([1,50]))

y3 = tf.matmul(y2_hat, W3) + b3

batch_y3 = tf.contrib.layers.batch_norm(y3, True)

y3_hat = tf.nn.relu(batch_y3)

# 출력층
W4 = tf.get_variable(name="W4", shape=[50,10], initializer = tf.contrib.layers.variance_scaling_initializer())
b4 = tf.Variable(tf.ones([1,10]))

z = tf.matmul(y3_hat,W4) + b4
z_hat = tf.nn.softmax(z)
y_predict = tf.argmax(z_hat, axis=1)

# 정확도 확인
y_onehot = tf.placeholder('float',[None,10]) # 정답 데이터를 담을 배열
y_label = tf.argmax(y_onehot, axis=1)

correction_prediction = tf.equal(y_predict, y_label)
accuracy = tf.reduce_mean(tf.cast(correction_prediction,'float'))

# 오차 확인
loss = -tf.reduce_sum(y_onehot * tf.log(z_hat+0.0000001), axis=1)
rs = tf.reduce_mean(loss)

# 학습
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# 변수 초기화
init = tf.global_variables_initializer()

# 그래프 실행
sess = tf.Session()
sess.run(init)

for i in range(1,601*20):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_x_test, batch_y_test = mnist.test.next_batch(100)
    sess.run(train, feed_dict={x:batch_xs, y_onehot:batch_ys})
    if not i % 600:
        print(i//600,"에폭 훈련데이터 정확도 : ",sess.run(accuracy, feed_dict={x:batch_xs, y_onehot:batch_ys}),"\t","테스트 데이터 정확도:", sess.run(accuracy, feed_dict={x:batch_x_test, y_onehot:batch_y_test}))


