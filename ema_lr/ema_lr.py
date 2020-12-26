# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/12/26 10:54 PM'

import tensorflow as tf
w1 = tf.Variable(0, dtype=tf.float32)
global_step = tf.Variable(0, dtype=tf.float32, trainable=False)
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
ema_op = ema.apply([w1])

init = tf.group(tf.global_variables_initializer(), tf.initialize_local_variables())
with tf.Session() as sess:
    sess.run(init)

    print('init w:', sess.run([w1, ema.average(w1)]))  # 用.average获得w1的滑动平均，也就是影子吧。
    sess.run(tf.assign(w1, 1))  # 手动修改w1的值
    sess.run(tf.assign(global_step, 1))
    sess.run(ema_op)  # 滑动一次。
    print('after an ema op')
    print('w:', sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)  # 滑动一次。
    print('after an ema op')
    print('w:', sess.run([w1, ema.average(w1)]))  # global_step不变动，不影响ema更新
    sess.run(ema_op)  # 滑动一次。
    print('after an ema op')
    print('w:', sess.run([w1, ema.average(w1)]))
    print('assign global_step:')
    # 假装进行了100轮迭代，w1变成10(其实ema没有更新中间那一百步）
    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)
    print('after 100 ema ops')
    print('w:', sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)


