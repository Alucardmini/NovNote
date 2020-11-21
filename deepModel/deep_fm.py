# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/11/21 11:47 PM'


from dataUtils.load_tiny_train_input import load_data


class DeepFM(object):
    def __init__(self, embed_dim: int, deep_layers: list, learning_rate: float):
        super(DeepFM, self).__init__

        self.embed_dim = embed_dim
        self.deep_layers = deep_layers
        self.learning_rate = learning_rate


    def build_model(self):
        pass


if __name__ == "__main__":
    pass