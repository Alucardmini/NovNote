# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/28 9:08 PM'


import tensorflow as tf

features = {
    'sex': ['male', 'male', 'female', 'female', 'mid', 'man'],
}

sex_tensor=tf.constant(['male', 'male', 'female', 'female', 'mid', 'man'], dtype=tf.string)

# sex_column = tf.feature_column.categorical_column_with_vocabulary_list('sex', ['male', 'female'])
sex_column = tf.string_to_hash_bucket(sex_tensor, 10)
# sex_column = tf.feature_column.indicator_column(sex_column)
# columns = [sex_column]
#
#
# inputs = tf.feature_column.input_layer(features, columns)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    sess.run(init)
    v = sess.run(sex_column)
    print(v)