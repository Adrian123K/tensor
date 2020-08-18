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

# 은닉1층
x = tf.placeholder('float',[None,784])
W1 = tf.get_variable(name="W1", shape=[784,100], initializer = tf.contrib.layers.variance_scaling_initializer())
b1 = tf.Variable(tf.ones([1,100]))

y = tf.matmul(x, W1) + b1
y_hat = tf.nn.relu(y)

# 은닉2층
W2 = tf.get_variable(name="W2", shape=[100,50], initializer = tf.contrib.layers.variance_scaling_initializer())
b2 = tf.Variable(tf.ones([1,50]))

y2 = tf.matmul(y_hat,W2) + b2
y2_hat = tf.nn.relu(y2)

# 출력층
W3 = tf.get_variable(name="W3", shape=[50,10], initializer = tf.contrib.layers.variance_scaling_initializer())
b3 = tf.Variable(tf.ones([1,10]))

z = tf.matmul(y2_hat,W3) + b3
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




