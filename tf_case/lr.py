# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2021/7/6 7:46 AM'


import tensorflow as tf


x = tf.random_normal([100, 1])
y = x * 0.5 + 0.7

random_init = tf.random_normal_initializer()
w = tf.get_variable(name="w", shape=[1, 1], initializer=random_init)
b = tf.get_variable(name="b", shape=[1, 1], initializer=random_init)

y_pred = tf.matmul(x, w) + b

init = tf.global_variables_initializer()

loss = tf.reduce_mean(tf.square(y_pred - y))

tf.summary.scalar("loss", loss)
tf.summary.histogram("w", w)
tf.summary.histogram("b", b)

merged = tf.summary.merge_all()

opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = opt.minimize(loss)


with tf.Session() as sess:

    sess.run(init)
    file_writer = tf.summary.FileWriter("./summary/linear", graph=sess.graph)

    for i in range(100):
        _, loss_val, w_val, b_val = sess.run([train_op, loss, w, b])

        print(loss_val, w_val, b_val)

        summary = sess.run(merged)
        file_writer.add_summary(summary, global_step=i)




