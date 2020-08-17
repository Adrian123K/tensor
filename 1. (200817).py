import tensorflow as tf
import warnings
import os
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
print('파이썬의 numpy 이용방법')
import numpy as np
a = np.array([0,0,1,0,0,0,0,0,0,0])
print(np.argmax(a, axis=0))

print('\n 텐서플로우 이용방법')
a = np.array([0,0,1,0,0,0,0,0,0,0])
b = tf.argmax(a, axis=0)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(b))

######################################################