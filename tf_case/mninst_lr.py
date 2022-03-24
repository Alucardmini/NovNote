# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/7/6 9:46 PM'

import keras.datasets.mnist as mnist
import tensorflow as tf
import numpy as np

import keras


(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


x = tf.placeholder(name="x", dtype=tf.float32, shape=[None, 28, 28])
y = tf.placeholder(name="y", dtype=tf.float32, shape=[None, 10])

x_data = tf.reshape(x, [-1, 28*28])
y_data = tf.reshape(y, [-1, 10])

random_init = tf.random_normal_initializer(mean=0.01, stddev=0.1)
w = tf.get_variable(name="w", shape=[28*28, 10], initializer=random_init)
b = tf.get_variable(name="b", shape=[10], initializer=random_init)

y_pred = tf.matmul(x_data, w) + b

mean, var = tf.nn.moments(y_pred, axes=[0])
scale = tf.Variable(tf.ones([10]))
shift = tf.Variable(tf.zeros([10]))  # scale 和 shift 又整理了输出
y_pred = tf.nn.batch_normalization(y_pred, mean=mean, variance=var, offset=shift, scale=scale, variance_epsilon=0.001)

out = tf.nn.softmax(y_pred)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=out))

arruracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y_pred, 1), tf.arg_max(y_data, 1)), tf.float32))

opt = tf.train.GradientDescentOptimizer(0.1)
train_op = opt.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(3000):

        _, loss_val = sess.run([train_op, loss], feed_dict={x: x_train, y: y_train})

        if i % 100 == 0:
            test_loss, arruracy_val = sess.run([loss, arruracy], feed_dict={x: x_test, y: y_test})
            print("step: ",i, " test_loss:", test_loss, " accuracy:",arruracy_val)