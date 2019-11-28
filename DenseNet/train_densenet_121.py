# -*- coding: utf-8 -*-
# @ File densenet_50.py
# @ Description
# @ Author alexchung
# @ Time 28/11/2019 PM 17:13

import os
from keras.models import Model
import keras.backend as K
from DenseNet.densenet_121 import DenseNet

pretrain_model = '/home/alex/Documents/pretraing_model/densenet/densenet121/densenet121_weights_tf.h5'

if __name__ == "__main__":
    input_shape = [224, 224, 3]
    num_classes = 1000
    num_filters = 64
    growth_k = 32
    num_dense_block = 4
    num_layers = [6, 12, 24, 16]
    reduction = 0.0
    keep_prob = 0.8
    weight_decay = 1e-4
    batch_norm_epsilon = 1e-5

    densenet_121 = DenseNet(input_shape=input_shape, num_classes=2, num_filters=num_filters, growth_k=growth_k,
                            num_dense_block=num_dense_block, num_layers=num_layers, reduction=reduction,
                            drop_rate=1.0 - keep_prob, weight_decay=weight_decay, batch_norm_epsilon=batch_norm_epsilon
                            ).inference()
    # densenet_121.load_weights(pretrain_model)
    print(densenet_121.summary())
