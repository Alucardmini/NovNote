# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/12/6 10:32 AM'

import tensorflow as tf
import sys
from dataUtils.load_tiny_train_input import load_data, get_batch


class FMModelArgs(object):
    deep_layers = [512, 256]
    embed_dim = 7
    field_size = 100
    epoch = 10
    feat_size = 100
    batch_size = 64
    learning_rate = 0.025
    is_train = False
    l2_reg_rate = 0.005
    checkpoint_dir = '../model'


class FMModel(object):
    def __init__(self, args):

        self.embed_dim = args.embed_dim
        self.learning_rate = args.learning_rate
        self.field_size = args.field_size
        self.feature_size = args.feat_size

        self.weights = self.init_weights()
        self.build_model()

    def init_weights(self):
        random_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        return {
            "first_factor_weight": tf.get_variable(name="first_factor_weight", shape=[self.feature_size, 1],
                                                   initializer=random_init),
            "embed_table": tf.get_variable(name="embed_table", shape=[self.feature_size, self.embed_dim],
                                           initializer=random_init)
        }

    def build_model(self):

        self.feat_value = tf.placeholder(name="feat_value", shape=[None, None], dtype=tf.float32)
        self.feat_index = tf.placeholder(name="feat_index", shape=[None, None], dtype=tf.int32)
        self.label = tf.placeholder(name="label", shape=[None, None], dtype=tf.float32)

        w1 = tf.nn.embedding_lookup(self.weights["first_factor_weight"], self.feat_index)
        wx = tf.reduce_sum(tf.multiply(w1, tf.reshape(self.feat_value, [-1, self.field_size, 1])), 2)

        vi = tf.nn.embedding_lookup(self.weights["embed_table"], self.feat_index)

        vixi = tf.multiply(vi, tf.reshape(self.feat_value, [-1, self.field_size, 1]))

        vixi_sum = tf.reduce_sum(vixi, axis=1)
        square_sum = tf.square(vixi_sum)
        vixi_square = tf.square(vixi)
        sum_square = tf.reduce_sum(vixi_square, axis=1)
        second_factor = 0.5 * tf.subtract(square_sum, sum_square)

        out = tf.concat([wx, second_factor], axis=1)
        self.out = tf.reduce_sum(out, axis=1)
        self.out = tf.reshape(self.out, [-1, 1])

        self.out = tf.nn.sigmoid(self.out)
        self.auc = tf.metrics.auc(tf.reshape(self.label, [-1, ]), tf.reshape(self.out, [-1, ]))

        self.loss = -tf.reduce_mean(
            self.label * tf.log(self.out + 1e-24) + (1 - self.label) * tf.log(1 - self.out + 1e-24))

        opt = tf.train.GradientDescentOptimizer(self.learning_rate)


        self.train_op = opt.minimize(self.loss)

    def train(self, sess, feat_index, feat_value, label):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value,
            self.label: label
        })
        return loss

    def predict(self, sess, feat_index, feat_value):
        result = sess.run([self.out], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value
        })
        return result

    def calc_auc(self, sess, feat_index, feat_value, labels):
        result = sess.run([self.auc], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value,
            self.label: labels
        })
        return result


if __name__ == "__main__":
    path = "../sourcedatas/tiny_train_input.csv"
    data = load_data(path)

    args = FMModelArgs()
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    args.feat_size = data['feat_dim']  # feature size
    args.field_size = len(data['xi'][0])  # 域名
    args.is_train = True

    with tf.Session(config=gpu_config) as sess:
        model = FMModel(args)
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
                    loss = model.train(sess, x_index, x_value, y)

                    if j % 100 == 0:
                        auc = model.calc_auc(sess, data['xi'], data['xv'], data['y_train'])
                        print("epoch %s  loss %s auc %s" % (i, loss, auc))

                        # import sklearn
                        # auc = sklearn.metrics.roc_auc_score(y, pred)

                        # print("epoch %s step %s, loss %s auc: %s" % (i, j, loss, auc))