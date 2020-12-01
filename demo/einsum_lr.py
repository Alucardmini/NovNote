# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/12/1 8:43 PM'


import tensorflow as tf
import numpy as np

random_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
a = tf.get_variable(name='a', shape=[16, 8, 5, 1, 128, 128], initializer=random_init)
b = tf.get_variable(name='b', shape=[16, 8, 1, 5, 128, 128], initializer=random_init)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    print(sess.run(tf.einsum('ijklno,ijlmno->ijkmno', a, b)).shape)
