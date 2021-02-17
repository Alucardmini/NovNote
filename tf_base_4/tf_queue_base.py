# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/2/17 8:16 PM'

import tensorflow as tf

q = tf.FIFOQueue(3, "float")
init = q.enqueue_many(([0.1, 0.2, 0.3], ))

x = q.dequeue()
y = x + 1.
q_inc = q.enqueue(y)

global_init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
init_op = tf.group(global_init, local_init)

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(init)

    for _ in range(2):
        sess.run(q_inc)

    for _ in range(3):
        print(sess.run(q.dequeue()))