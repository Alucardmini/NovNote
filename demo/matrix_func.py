# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/28 11:17 PM'


import tensorflow as tf

init = tf.global_variables_initializer()

x = tf.fill([2, 3], 5.0)
y = tf.fill([2, 3], 2.0)

z = tf.constant([[i*3 + j for j in range(5)] for i in range(3)])

with tf.Session() as sess:
    sess.run(init)

    # print(sess.run(x))
    # print(sess.run(y))
    #
    # # print(sess.run(tf.matmul(x, y)))
    # print(sess.run(tf.multiply(x, y)))
    #
    # print(sess.run(tf.reduce_sum(x, 0)))
    # print(sess.run(tf.reduce_sum(x, 1)))

    print(sess.run(z))

    print(sess.run(tf.tile(z, [1, 2])))

    print(sess.run(tf.slice(tf.tile(z, [1, 2]), [0, 2], [2, 3])))
