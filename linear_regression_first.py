import tensorflow as tf
import numpy as np

'''
用如下参数时，很快出现溢出现象

feature_count = 1000
sample_count = 100000
'''

feature_count = 1000
sample_count = 100000

x_data = np.float32(np.random.rand(feature_count, sample_count))
w = np.linspace(-1, 2, 1000)
y_data = np.dot(w, x_data) + 0.3

#genrate linear model
b = tf.Variable(tf.zeros(1), dtype=tf.float32)
W = tf.Variable(tf.random_uniform([1,1000], -2., 2.), dtype=tf.float32)
y = tf.matmul(W, x_data) + b

#minize square sum
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#init variable
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(0, 2000):
    sess.run(train)
    if step % 200 == 0:
        print step,sess.run(W), sess.run(b)
