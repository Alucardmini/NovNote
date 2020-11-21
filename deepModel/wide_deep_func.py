# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/22 1:12 AM'

import tensorflow as tf
import sys
import numpy as np
from dataUtils.load_tiny_train_input import load_data

tf.logging.set_verbosity(tf.logging.INFO)

def init_weights(field_size, feature_size, deep_layers):
    random_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    merge_layer_size = field_size + deep_layers[-1]

    weights = {
        "wide_weights": tf.get_variable(name="wide_weights", shape=[feature_size, 1],
                                        initializer=random_init),

        "layer_0": tf.get_variable(name="layer_0",
                                   shape=[field_size, deep_layers[0]],
                                   initializer=random_init),
        "bias_0": tf.get_variable(name="bias_0", shape=[1, deep_layers[0]], initializer=random_init),
        "merge_layer": tf.get_variable(name="merge_layer", shape=[merge_layer_size, 1], initializer=random_init),
        "merge_bias": tf.get_variable(name="merge_bias", shape=[1], initializer=random_init)
    }
    cnt_hidden_layer = len(deep_layers)
    if cnt_hidden_layer > 0:
        for i in range(1, cnt_hidden_layer):
            weights["layer_" + str(i)] = tf.get_variable(name="layer_" + str(i),
                                                              shape=[deep_layers[i - 1], deep_layers[i]],
                                                              initializer=random_init)
            weights["bias_" + str(i)] = tf.get_variable(name="bias_" + str(i),
                                                             shape=[1, deep_layers[i]],
                                                             initializer=random_init)

    return weights


def model_fn(features, labels, mode, params):
    feat_value = features['feat_value']
    feat_index = features['feat_index']

    field_size = params['field_size']
    feature_size = params['feature_size']
    deep_layers = params['deep_layers']
    deep_activation = params['deep_activation']
    l2_reg_rate = params['l2_reg_rate']
    learning_rate = params['learning_rate']

    weights = init_weights(field_size, feature_size, deep_layers)

    #  wide part
    wide_part = tf.multiply(tf.nn.embedding_lookup(weights["wide_weights"], feat_index),
                            tf.reshape(feat_value, [-1, field_size, 1]))
    wide_part = tf.reduce_sum(wide_part, axis=2)

    # DEEP PART
    deep_input_size = field_size
    cnt_hidden_layer = len(deep_layers)
    # DNN
    deep_part = tf.reshape(feat_value, [-1, deep_input_size])
    for i in range(0, cnt_hidden_layer):
        deep_part = tf.add(tf.matmul(deep_part, weights["layer_" + str(i)]),
                                weights["bias_" + str(i)])
        deep_part = deep_activation(deep_part)
    out = tf.add(tf.matmul(tf.concat([wide_part, deep_part], axis=1),
                                weights["merge_layer"]), weights["merge_bias"])
    out = tf.nn.sigmoid(out)

    loss = -tf.reduce_mean(
        labels * tf.log(out + 1e-24) + (1 - labels) * tf.log(1 - out + 1e-24))
    loss += tf.contrib.layers.l2_regularizer(l2_reg_rate)(weights["merge_layer"])
    for i in range(len(deep_layers)):
        loss += tf.contrib.layers.l2_regularizer(l2_reg_rate)(weights["layer_%d" % i])

    global_step = tf.Variable(0, trainable=False)
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    trainable_params = tf.trainable_variables()

    gradients = tf.gradients(loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)

    train_op = opt.apply_gradients(
        zip(clip_gradients, trainable_params), global_step=global_step)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(tf.arg_max(out, 1), labels),
                       "auc": tf.metrics.auc(tf.reshape(labels, [-1, ]), tf.reshape(out, [-1, ]))}

    return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, eval_metric_ops=eval_metric_ops, loss=loss)


if __name__ == "__main__":
    path = "../sourcedatas/tiny_train_input.csv"
    data = load_data(path)
    model_params = {
                    'deep_layers': [512, 256],
                    'feature_size': data['feat_dim'],  # feature size
                    'field_size': len(data['xi'][0]),  # 域名
                    'learning_rate': 0.05,
                    'l2_reg_rate': 0.005,
                    'deep_activation': tf.nn.relu
                    }
    estimator = tf.estimator.Estimator(model_fn=model_fn, params=model_params, model_dir="model/wide_deep")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"feat_value": np.array(data['xv']).astype(np.float32),
                                                           "feat_index": np.array(data['xi'])},
                                                        y=np.array(data['y_train']).astype(np.float32),
                                                        num_epochs=2,
                                                        batch_size=128,
                                                        shuffle=True)

    estimator.train(input_fn=train_input_fn, steps=3000)
