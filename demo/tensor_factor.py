# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/29 10:45 AM'


import tensorflow as tf
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 新建一个固定张量

_0_s = tf.zeros(shape=[2, 3, 4])
_1_s = tf.ones(shape=[4, 5])
_x_s = tf.fill([2, 3, 4], 10)

# array 转张量
_array_s = tf.constant([[i*4 + j for j in range(4)] for i in range(4)])


_add_1_s = tf.fill([2, 3], 10)
_add_2_s = tf.fill([2, 3], 10)
_add_3_s = tf.fill([2, 1], 10)
_add_4_s = tf.fill([1, 3], 10)
# tf.add
# print(sess.run(tf.add(_add_1_s, _add_2_s)))
# print(sess.run(_add_1_s + _add_2_s))
# print(sess.run(_add_1_s + _add_3_s))
# print(sess.run(tf.add(_add_1_s, _add_4_s)))


# tf.matmul 矩阵相乘 M*N 要求M最后一个个维度和N导数第二个维度值相同， 且M 和 N除去最后两个维度以外 其他维度数值相同

# x = tf.random_normal([3, 4])
# w = tf.random_normal([4, 3])
# print(tf.matmul(x, w).shape)

# x = tf.fill([5, 2, 3, 4], 3)

# b = tf.fill([1], 1)
# print(sess.run(tf.add(x, b)))

# w = tf.fill([5, 2, 4, 6], 5)
# print(tf.matmul(x, w).shape)
# print(sess.run(tf.matmul(x, w)))

# # tf.multiply 元素相乘 以下结果都是 shape = [5, 2, 3, 4] 元素数值都为6的 tensor
# x = tf.fill([5, 2, 3, 4], 3)
# b = tf.fill([1], 2)
# print(sess.run(x * b))
#
# z = tf.fill([5, 2, 1, 1], 2)
# print(sess.run(x * z))


# # tf.tile
# x = tf.get_variable(name='x', shape=[2, 3, 4], initializer=tf.random_normal_initializer(mean=0.01, stddev=0.01))
# # 复制元素 最后输出一个 [2 * 6, 3* 4, 4* 3]的tensor
# z = tf.tile(x, [6, 4, 3])
# print(sess.run(tf.shape(z)))
#
# # 复制元素 最后输出一个 [2 * 1, 3*2, 4* 3]的tensor
# z = tf.tile(x, [1, 2, 3])
# print(sess.run(tf.shape(z)))


# tf.slice 不太好解释， 定一个起点 在定义一个size 进行连续性切割
# import numpy as np
# x = tf.convert_to_tensor(np.random.randint(0, 100, size=[4, 5, 6]))
# print(x.shape, "\n",sess.run(x))
#
# z = tf.slice(x, [0, 1, 2], [4, 4, 2])
# print("slice ----", z.shape)
# print(sess.run(z))

# tf.squeeze
import numpy as np
# x = tf.convert_to_tensor(np.random.randint(0, 100, size=[4, 5, 1, 5, 2, 1, 5, 1]))
# print(tf.squeeze(x).shape)


# tf.concat 输出拼接好的 tensor 下方例子中shape=(2, 3, 6) 要求其他维度值相同
# x = tf.convert_to_tensor(np.random.randint(0, 100, size=[2, 3, 4]))
# y = tf.convert_to_tensor(np.random.randint(0, 100, size=[2, 3, 2]))
# print(tf.concat([x, y], axis=2).shape)

# tf.gather

# x = tf.convert_to_tensor(np.random.randint(0, 100, size=[2, 3, 4]))
#
# # print(tf.gather(x, [[0, 1, 2], [2, 2, 2]]).shape)
# # print(tf.gather(x, [1, 2, 3]).shape)
# # print(sess.run(tf.gather(x, [1, 2, 3])))
#
# temp = tf.range(0,10)*10 + tf.constant(1,shape=[10])
# temp2 = tf.gather(temp,[1,5,9])
#
# print(sess.run(temp2))
# print(temp2.shape)


# tf.stack
# tf.matrix_inverse_
# tf.expand_dim_
