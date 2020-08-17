import tensorflow as tf
import numpy as np
import warnings
import os
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)


# hello = tf.constant("Hello Tensorflow")
#
# sess = tf.Session() # 그래프를 실행할 세션을 구성
#
# print(sess.run(hello))
# print(str(sess.run(hello), encoding='utf-8'))

# x = tf.constant(35, name='x') # x라는 상수값을 만들고 숫자 35를 지정
# y = tf.Variable(x+5, name='y') # y라는 변수를 만들고 방정식 x+5를 정의
# model = tf.global_variables_initializer() # 변수를 생성했으면 초기화 해줘야하는데 필요한 명령어. 무조건 초기화를 해줘야 실행 가능
#
# sess = tf.Session() # 그래프를 실행할 세션을 구성
# sess.run(model) # 변수를 초기화 하겠다고 정의한 model을 실행
# print(sess.run(y))

# a = tf.constant(10)
# b = tf.constant(32)
# c = tf.add(a,b)
# print(c)
#
# sess = tf.Session()
# print(sess.run(c))
# sess.close()

# with tf.Session() as sess:
#     print(sess.run(c))

# a = tf.add(1, 2)
# b = tf.multiply(a, 3)
# c = tf.add(b, 5)
# d = tf.multiply(c, 6)
# e = tf.multiply(d, 5)
# f = tf.div(e, 6)
# g = tf.add(f, d)
# h = tf.multiply(g, f)
#
# with tf.Session() as sess:
#     print(sess.run(h))

# x = tf.Variable(0)
# init = tf.global_variables_initializer() # 변수를 생성했으면 초기화를 해줘야 함
#
# with tf.Session() as sess :
#     sess.run(init)
#     for i in range(5):
#         x = x+1
#         print(sess.run(x))

#112 구구단 2단까지
# with tf.Session() as sess:
#     for i in range(1,10):
#         print('2 x', i ,'=',sess.run(tf.multiply(2,i)))

# 113 구구한 9단까지
# with tf.Session() as sess:
#     for i in range(2,10):
#         for j in range(1,10):
#             print(i,'x',j, '=',sess.run(tf.multiply(i,j)))

# x = tf.Variable(0)
# y = tf.Variable(0)
# z = tf.multiply(x, y)
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     for i in range(2,10):
#         for j in range(1,10):
#             print(i, 'x', j, '=', sess.run(z, feed_dict=({x:i, y:j})))

###################################################### bias 구현
# print('numpy 구현')
# import numpy as np
# a = np.zeros((2,2))
# b = np.ones((2,2))
# print(a)
# print(b)
#
# print('\n tensorflow 구현')
# a = tf.zeros((2,2))
# b = tf.ones((2,2))
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(a))
#     print(sess.run(b))

###################################################### 배열 최댓값 인덱스 찾기
# print('파이썬의 numpy 이용방법')
# import numpy as np
# a = np.array([0,0,1,0,0,0,0,0,0,0])
# print(np.argmax(a, axis=0))
#
# print('\n 텐서플로우 이용방법')
# a = np.array([0,0,1,0,0,0,0,0,0,0])
# b = tf.argmax(a, axis=0)
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(b))

#114 배열에서 최댓값 갖는 인덱스 찾기
# import numpy as np
# a = np.array([[0,0,1,0,0,0,0,0,0,0],
#               [0,0,0,0,1,0,0,0,0,0],
#               [0,0,0,0,0,0,1,0,0,0],
#               [1,0,0,0,0,0,0,0,0,0],
#               [0,0,0,0,0,0,0,0,1,0]])
# import tensorflow as tf
# b = tf.argmax(a, axis=1) # 0은 열, 1은 행
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(b))

###################################################### 다차원 배열에서 행의 토탈값 출력
# import numpy as np
# a = np.array([[[1, 2, 3],
#                [2, 1, 4],
#                [5, 2, 1],
#                [6, 3, 2]],
#               [[5, 1, 3],
#                [1, 3, 4],
#                [4, 2, 6],
#                [3, 9, 3]],
#               [[4, 5, 6],
#                [7, 4, 3],
#                [2, 1, 5],
#                [4, 3, 1]]])
#
# print('numpy 이용했을 때')
# print(np.sum(a, axis=0))
#
# print('\n 텐서 플로우 이용했을 때')
# a = np.array([[[1, 2, 3],
#                [2, 1, 4],
#                [5, 2, 1],
#                [6, 3, 2]],
#               [[5, 1, 3],
#                [1, 3, 4],
#                [4, 2, 6],
#                [3, 9, 3]],
#               [[4, 5, 6],
#                [7, 4, 3],
#                [2, 1, 5],
#                [4, 3, 1]]])
#
# d = tf.reduce_sum(a, reduction_indices=[0])
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(d))

# 115 다차원 평균값 구하기
# a = np.array([[[1, 2, 3],
#                [2, 1, 4],
#                [5, 2, 1],
#                [6, 3, 2]],
#               [[5, 1, 3],
#                [1, 3, 4],
#                [4, 2, 6],
#                [3, 9, 3]],
#               [[4, 5, 6],
#                [7, 4, 3],
#                [2, 1, 5],
#                [4, 3, 1]]])
#
# d = tf.reduce_mean(a, reduction_indices=[0])
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(d))

###################################################### 텐서플로우 내적
# print('numpy 형식')
# a = np.array([[2,2,2],[2,2,2]])
# b = np.array([[3,3],[3,3],[3,3]])
# res = np.dot(a,b)
# print(res)
#
# print('\n tensorflow 형식')
# x = tf.placeholder("float",[2,3]) # (2,3) 행렬의 실수형 데이터를 담을 변수
# y = tf.placeholder("float",[3,2])
# rs = tf.matmul(x, y)
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(rs, feed_dict={x:[[2,2,2],[2,2,2]], y:[[3,3],[3,3],[3,3]]}))

# 116
# a = np.array([[6,7],[3,5],[2,9]])
# b = np.array([[3,8,1],[4,5,2]])
#
# import tensorflow as tf
# x = tf.placeholder("float",[3,2]) # (2,3) 행렬의 실수형 데이터를 담을 변수
# y = tf.placeholder("float",[2,3])
# rs = tf.matmul(x, y)
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(rs, feed_dict={x:[[6,7],[3,5],[2,9]], y:[[3,8,1],[4,5,2]]}))

###################################################### 텐서플로우에서 mnist 불러오기
# mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
#
# batch_xs, batch_ys = mnist.train.next_batch(100)
#
# # print(batch_xs.shape)
# # print(batch_ys.shape)
#
# # 117
# batch_xs, batch_ys = mnist.test.next_batch(100)
#
# print(batch_xs.shape)
# print(batch_ys.shape)
#
# ###################################################### 배치단위 데이터 불러오기
# x = tf.placeholder("float",[None,3]) # None으로 작성 -> 행의 개수 무관하게 작성 가능
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
#
# sess.run(init)
# print(sess.run(x, feed_dict={x:[[2,2,2],[2,2,2],[2,2,2]]}))
#
# sess.close()

# 118
# mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
#
# x = tf.placeholder('float',[None,784])
#
# batch_xs, batch_ys = mnist.train.next_batch(100)
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(x, feed_dict={x:batch_xs}).shape)

###################################################### 텐서 플로우로 신경망 만들 때 가중치 생성 코드
# W1 = tf.Variable(tf.random_uniform([784,50],-1,1)) # [-1,1] 사이의 숫자로 (784,50) 행렬의 변수를 W1로 생성
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(W1).shape)
#     print(sess.run(W1))

# 119
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# x = tf.placeholder('float', [None, 784])
# batch_xs, batch_ys = mnist.train.next_batch(100)
#
# W1 = tf.Variable(tf.random_uniform([784, 50], -1, 1))
# init = tf.global_variables_initializer()
#
# rs = tf.matmul(x, W1)
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(rs, feed_dict={x: batch_xs}))
#     print(sess.run(rs, feed_dict={x: batch_xs}).shape)

# 121
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# x = tf.placeholder('float', [None, 784])
#
# batch_xs, batch_ys = mnist.train.next_batch(100)
#
# W1 = tf.Variable(tf.random_uniform([784, 50], -1, 1))
# b = tf.Variable(tf.ones([1, 50]))
# rs = tf.matmul(x, W1) + b
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(rs, feed_dict={x: batch_xs}))

# 122
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder('float', [None, 784])

batch_xs, batch_ys = mnist.train.next_batch(100)

W1 = tf.Variable(tf.random_uniform([784, 50], -1, 1))
b = tf.Variable(tf.ones([1, 50]))
rs = tf.matmul(x, W1) + b

y_hat = tf.sigmoid(rs)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(y_hat, feed_dict={x: batch_xs}))

######################################################
