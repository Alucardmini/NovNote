# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/12/26 11:08 PM'


# 同样的例子，修改一下，让w1动态变化，ema在后边追。
import tensorflow as tf

w1 = tf.Variable(0, dtype=tf.float32)
global_step = tf.Variable(0, dtype=tf.float32, trainable=False)  # 不会被ema做平均
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

ema_op = ema.apply(tf.trainable_variables())
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('init w:', sess.run([w1, ema.average(w1)]))  # 用.average获得w1的滑动平均，也就是影子吧。
    sess.run(tf.assign(w1, 1))  # 手动修改w1的值
    sess.run(tf.assign(global_step, 1))
    sess.run(ema_op)  # 滑动一次。

    print('after an ema op')
    print('w:', sess.run([w1, ema.average(w1)]))
    # 假装进行了100轮迭代，w1变成10(其实ema没有更新中间那一百步）
    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)
    print('after 100 ema ops')
    print('w:', sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)

    # 再拿同样的w=10多更新几次影子,让影子逼近w1，同时，w1也变化。
    for i in range(100):
        sess.run(tf.assign_add(w1, 1))
        sess.run(ema_op)
        if i % 10 == 0:
            print('w:', sess.run([w1, ema.average(w1)]))
            print('global_step:', sess.run(global_step))
            #             print('global_step ema:',sess.run([global_step,ema.average(global_step)]))#global_step的ema