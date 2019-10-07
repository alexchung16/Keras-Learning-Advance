#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File layer_weight_sharing.py
# @ Description
# @ Author alexchung
# @ Time 7/10/2019 PM 11:41

import os
import numpy as np
from keras import Input
from keras.models import Model
from keras import layers


def layerWeightSharing():
    """
    layer weight sharing
    :return:
    """

    lstm = layers.LSTM(32)

    left_input = Input(shape=(None, 128))
    left_output = lstm(inputs=left_input)

    right_input = Input(shape=(None, 128))
    right_output = lstm(inputs=right_input)

    merge = layers.concatenate(inputs=[left_output, right_output], axis=-1)
    predict = layers.Dense(units=1, activation='sigmoid')(merge)
    model = Model(inputs=[left_input, right_input], outputs=predict)

    return model


if __name__ == "__main__":
    model = layerWeightSharing()
    print(model.summary())