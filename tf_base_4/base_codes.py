# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/2/17 7:27 PM'

import tensorflow as tf

random_init = tf.random_normal_initializer(0.0, 0.1)

with tf.variable_scope("foo"):
    with tf.name_scope("bar"):
        t = tf.get_variable(name="t", shape=[100, 20], initializer=random_init, dtype=tf.float32)


fc_mean, fc_var = tf.nn.moments(t, axes=[0])

scale = tf.Variable(tf.ones([20]))
shift = tf.Variable(tf.ones([20]))
epsilon = 10e-3

k = tf.nn.batch_normalization(t, fc_mean, fc_var, shift, scale, epsilon)


global_init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()

init = tf.group(global_init, local_init)

with tf.Session() as sess:
    sess.run(init)

    print(sess.run(k))
    print(sess.run(t))



