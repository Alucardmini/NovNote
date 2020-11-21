# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/21 11:50 PM'
import tensorflow as tf
import sys
from dataUtils.load_tiny_train_input import load_data, get_batch


class WideDeepArgs(object):
    feat_size = 100
    deep_layers = [512, 256]
    field_size = 100
    epoch = 10
    batch_size = 64
    learning_rate = 0.05
    is_train = False
    l2_reg_rate = 0.005
    checkpoint_dir = '../model'


class WideDeep(object):
    def __init__(self, args:WideDeepArgs):
        self.deep_layers = args.deep_layers
        self.l2_reg_rate = args.l2_reg_rate
        self.learning_rate = args.learning_rate
        self.feature_size = args.feat_size
        self.field_size = args.field_size
        self.deep_activation = tf.nn.relu
        self.weights = dict()
        self.init_weights()
        self.build_model()

    def init_weights(self):
        random_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        merge_layer_size = self.field_size + self.deep_layers[-1]

        self.weights = {
            "wide_weights": tf.get_variable(name="wide_weights", shape=[self.feature_size, 1],
                                                   initializer=random_init),

            "layer_0": tf.get_variable(name="layer_0",
                                       shape=[self.field_size, self.deep_layers[0]],
                                       initializer=random_init),
            "bias_0": tf.get_variable(name="bias_0", shape=[1, self.deep_layers[0]], initializer=random_init),
            "merge_layer": tf.get_variable(name="merge_layer", shape=[merge_layer_size, 1], initializer=random_init),
            "merge_bias": tf.get_variable(name="merge_bias", shape=[1], initializer=random_init)
        }
        cnt_hidden_layer = len(self.deep_layers)
        if cnt_hidden_layer > 0:
            for i in range(1, cnt_hidden_layer):
                self.weights["layer_" + str(i)] = tf.get_variable(name="layer_" + str(i),
                                                                  shape=[self.deep_layers[i - 1], self.deep_layers[i]],
                                                                  initializer=random_init)
                self.weights["bias_" + str(i)] = tf.get_variable(name="bias_" + str(i),
                                                                 shape=[1, self.deep_layers[i]],
                                                                 initializer=random_init)

    def build_model(self):

        self.feat_index = tf.placeholder(tf.int32, [None, None], name="feat_index")
        self.feat_value = tf.placeholder(tf.float32, [None, None], name="feat_value")
        self.label = tf.placeholder(tf.float32, [None, None], name="label")

        #  wide part
        self.wide_part = tf.multiply(tf.nn.embedding_lookup(self.weights["wide_weights"], self.feat_index),
                                        tf.reshape(self.feat_value, [-1, self.field_size, 1]))
        self.wide_part = tf.reduce_sum(self.wide_part, axis=2)

        # DEEP PART
        deep_input_size = self.field_size
        cnt_hidden_layer = len(self.deep_layers)
        # DNN
        self.deep_part = tf.reshape(self.feat_value, [-1, deep_input_size])
        for i in range(0, cnt_hidden_layer):
            self.deep_part = tf.add(tf.matmul(self.deep_part, self.weights["layer_" + str(i)]),
                                         self.weights["bias_" + str(i)])
            self.deep_part = self.deep_activation(self.deep_part)
        self.out = tf.add(tf.matmul(tf.concat([self.wide_part, self.deep_part], axis=1),
                                    self.weights["merge_layer"]), self.weights["merge_bias"])
        self.out = tf.nn.sigmoid(self.out)

        self.auc = tf.metrics.auc(tf.reshape(self.label, [-1, ]), tf.reshape(self.out, [-1, ]))

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

    def calc_auc(self, sess, feat_index, feat_value, labels):
        result = sess.run([self.auc], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value,
            self.label: labels
        })
        return result


    def save(self, sess, path):
        tf.train.Saver.save(sess, save_path=path)

    def restore(self, sess, path):
        tf.train.Saver.restore(sess, save_path=path)


if __name__ == "__main__":
    path = "../sourcedatas/tiny_train_input.csv"
    data = load_data(path)

    args = WideDeepArgs()
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    args.feat_size = data['feat_dim']  # feature size
    args.field_size = len(data['xi'][0])  # 域名
    args.is_train = True

    with tf.Session(config=gpu_config) as sess:
        model = WideDeep(args)
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

                        # pred = model.predict(sess, data['xi'], data['xv'])[0]
                        auc = model.calc_auc(sess, data['xi'], data['xv'], data['y_train'])

                        # import sklearn
                        # auc = sklearn.metrics.roc_auc_score(y, pred)

                        print("epoch %s step %s, loss %s auc: %s" % (i, j, loss, auc))



