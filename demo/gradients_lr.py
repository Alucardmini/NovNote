# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/30 10:24 PM'

import tensorflow as tf

x = tf.Variable(initial_value=50., dtype='float32')
w = tf.Variable(initial_value=10., dtype='float32')
y = w*x

opt = tf.train.GradientDescentOptimizer(0.1)
grad = opt.compute_gradients(y, [w])
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    print(sess.run(grad))


def minimize():

    grads_and_vars = opt.compute_gradients(y, [w])
    vars_with_grad = [v for g, v in grads_and_vars if g is not None]

    opt.apply_gradients(grads_and_vars, )
