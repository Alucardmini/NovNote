# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/21 3:09 PM'
import pandas as pd

import tensorflow as tf
import numpy as np
import sys


class ModelArgs(object):
    feat_size = 100
    embed_size = 100
    deep_layers = [512, 256, 128]
    field_size = 100
    epoch = 3
    batch_size = 64
    learning_rate = 0.05
    is_train = False
    l2_reg_rate = 0.01
    checkpoint_dir = '../model'


class DeepFM(object):

    def __init__(self, args: ModelArgs):
        super(DeepFM, self).__init__()
        self.feat_embed_dim = args.embed_size
        self.deep_layers = args.deep_layers
        self.l2_reg_rate = args.l2_reg_rate
        self.learning_rate = args.learning_rate
        self.feature_size = args.feat_size
        self.field_size = args.field_size
        self.deep_activation = tf.nn.relu

        self.weights = dict()
        self.build_model()

    def build_model(self):

        self.feat_index = tf.placeholder(tf.int32, [None, None], name="feat_index")
        self.feat_value = tf.placeholder(tf.float32, [None, None], name="feat_value")
        self.label = tf.placeholder(tf.float32, [None, None], name="label")

        random_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        merge_layer_size = self.field_size + self.feat_embed_dim + self.deep_layers[-1]

        self.weights = {
            "first_factor_weight": tf.get_variable(name="first_factor_weight", shape=[self.feature_size, 1], initializer=random_init),
            "embedding_weight": tf.get_variable(name="embedding_weight", shape=[self.feature_size, self.feat_embed_dim], initializer=random_init),
            "layer_0": tf.get_variable(name="layer_0", shape=[self.feat_embed_dim * self.field_size, self.deep_layers[0]], initializer=random_init),
            "bias_0": tf.get_variable(name="bias_0", shape=[1, self.deep_layers[0]], initializer=random_init),
            "merge_layer": tf.get_variable(name="merge_layer", shape=[merge_layer_size, 1], initializer=random_init),
            "merge_bias": tf.get_variable(name="merge_bias", shape=[1], initializer=random_init)
        }
        cnt_hidden_layer = len(self.deep_layers)
        if cnt_hidden_layer > 0:
            for i in range(1, cnt_hidden_layer):
                self.weights["layer_" + str(i)] = tf.get_variable(name="layer_" + str(i),
                                                                  shape=[self.deep_layers[i-1], self.deep_layers[i]],
                                                                  initializer=random_init)
                self.weights["bias_" + str(i)] = tf.get_variable(name="bias_" + str(i),
                                                                 shape=[1, self.deep_layers[i]],
                                                                 initializer=random_init)

        #  X*W
        self.first_factor = tf.multiply(tf.nn.embedding_lookup(self.weights["first_factor_weight"], self.feat_index),
                                        tf.reshape(self.feat_value, [-1, self.field_size, 1]))
        self.first_factor = tf.reduce_sum(self.first_factor, axis=2)
        self.embed_part_weight = tf.nn.embedding_lookup(self.weights["embedding_weight"], self.feat_index)
        tmp = tf.reshape(self.feat_value, [-1, self.field_size, 1])
        # FM
        self.embed_part = tf.multiply(self.embed_part_weight, tmp)
        self.second_factor_sum_square = tf.square(tf.reduce_sum(self.embed_part, 1))
        self.second_factor_square_sum = tf.reduce_sum(tf.square(self.embed_part), 1)
        self.second_factor = 0.5 * tf.subtract(self.second_factor_sum_square, self.second_factor_square_sum)
        deep_input_size = self.feat_embed_dim * self.field_size
        cnt_hidden_layer = len(self.deep_layers)
        # DNN
        self.deep_embedding = tf.reshape(self.embed_part, [-1, deep_input_size])
        for i in range(0, cnt_hidden_layer):
            self.deep_embedding = tf.add(tf.matmul(self.deep_embedding, self.weights["layer_" + str(i)]),
                                         self.weights["bias_" + str(i)])
            self.deep_embedding = self.deep_activation(self.deep_embedding)
        self.fm_part = tf.concat([self.first_factor, self.second_factor], axis=1)
        self.out = tf.add(tf.matmul(tf.concat([self.fm_part, self.deep_embedding], axis=1),
                                    self.weights["merge_layer"]), self.weights["merge_bias"])
        self.out = tf.nn.sigmoid(self.out)
        self.loss = -tf.reduce_mean(
            self.label * tf.log(self.out + 1e-24) + (1 - self.label) * tf.log(1 - self.out + 1e-24))
        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weights["merge_layer"])
        for i in range(len(self.deep_layers)):
            self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weights["layer_%d" % i])

        self.global_step = tf.Variable(0, trainable=False)
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        trainable_params = tf.trainable_variables()

        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)

        self.train_op = opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, feat_index, feat_value, label):
        loss, _, step = sess.run([self.loss, self.train_op, self.global_step], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value,
            self.label: label
        })
        return loss, step

    def predict(self, sess, feat_index, feat_value):

        result = sess.run([self.out], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value
        })
        return result

    def save(self, sess, path):
        tf.train.Saver.save(sess, save_path=path)

    def restore(self, sess, path):
        tf.train.Saver.restore(sess, save_path=path)


def get_batch(xi, xv, y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)

    return xi[start: end], xv[start: end], np.array(y[start: end])


def load_data(path: str)-> dict:

    src_df = pd.read_csv(path, header=None)
    src_df.columns = ['c' + str(i) for i in range(src_df.shape[1])]
    labels = src_df.c0.values
    labels = labels.reshape(len(labels), 1)

    discrete_feats = pd.DataFrame()   # 离散特征
    continuous_feats = pd.DataFrame()  # 连续特征 进行正态归一化

    discrete_col_names = []
    continuous_col_names = []
    feat_index = 1
    feat_dict = {}

    for i in range(1, src_df.shape[1]):
        temp_data = src_df.iloc[:, i]
        uniqu_val = temp_data.unique()
        val_length = len(uniqu_val)
        col_name = temp_data.name

        if val_length > 10:
            # 连续特征
            temp_data = (temp_data - temp_data.mean()) / temp_data.std()
            continuous_col_names.append(col_name)
            continuous_feats = pd.concat([continuous_feats, temp_data], axis=1)
            feat_dict[col_name] = feat_index
            feat_index += 1
        else:
            # 离散特征
            discrete_col_names.append(col_name)
            discrete_feats = pd.concat([discrete_feats, temp_data], axis=1)
            feat_dict[col_name] = dict(zip(uniqu_val, range(feat_index, val_length + feat_index)))
            feat_index += val_length

    feature_values_df = pd.concat([continuous_feats, discrete_feats], axis=1)
    feature_indices_df = feature_values_df.copy()

    for i in feature_indices_df.columns:
        if i in continuous_feats:
            feature_indices_df[i] = feat_dict[i]
        else:
            feature_indices_df[i] = feature_indices_df[i].map(feat_dict[i])
            feature_indices_df[i] = 1

    train_data = {}
    train_data['y_train'] = labels

    train_data['xi'] = feature_indices_df.values.tolist()
    train_data['xv'] = feature_values_df.values.tolist()
    train_data['feat_dim'] = feat_index
    return train_data


if __name__ == "__main__":
    path = "../sourcedatas/tiny_train_input.csv"
    data = load_data(path)

    args = ModelArgs()
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    args.feat_size = data['feat_dim']  # feature size
    args.field_size = len(data['xi'][0])  # 域名
    args.is_train = True

    with tf.Session(config=gpu_config) as sess:
        model = DeepFM(args)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        cnt = int(len(data['y_train']) / args.batch_size)
        print('time all %s' % cnt)
        sys.stdout.flush()

        if args.is_train:
            for i in range(args.epoch):
                print("epoch %s:" % i)
                for j in range(0, cnt):
                    x_index, x_value, y = get_batch(data['xi'], data['xv'], data['y_train'], args.batch_size, j)
                    loss, step = model.train(sess, x_index, x_value, y)

                    if j % 100 == 0:
                        print("epoch %s step %s, loss %s" % (i, j, loss))