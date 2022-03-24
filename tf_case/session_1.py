# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/7/11 9:41 AM'


import tensorflow as tf

import numpy as np

x_data = np.random.rand(100, 1)


noise = np.random.uniform(0.01, 0.02)

y_data = x_data * 0.8 + 0.7 + noise
# y_data = x_data * 0.8 + 0.7


x = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="x")
y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y")

random_init = tf.random_normal_initializer()
w = tf.get_variable(shape=[1, 1], dtype=tf.float32, initializer=random_init, name="w")
b = tf.get_variable(shape=[1, 1], dtype=tf.float32, initializer=random_init, name="b")

y_pred = x*w + b

loss = tf.reduce_mean(tf.square(y_pred - y))

opt = tf.train.GradientDescentOptimizer(0.1)

train_op = opt.minimize(loss)

init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):

        _,loss_val, w_val, b_val = sess.run([train_op, loss, w, b], feed_dict={x: x_data, y: y_data})

        print(i,loss_val, w_val, b_val)