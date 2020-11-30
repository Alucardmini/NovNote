# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/29 9:09 PM'

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
sess = tf.Session()

saver = tf.train.import_meta_graph("../deepModel/model/wide_deep/model.ckpt-0.meta")
saver.restore(sess,  tf.train.latest_checkpoint("../deepModel/model/wide_deep/"))


checkpoint_path = "../deepModel/model/wide_deep/model.ckpt-0"
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)


var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key).shape)
