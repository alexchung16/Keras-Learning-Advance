#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File test
# @ Description
# @ Author alexchung
# @ Time 24/9/2019 PM 17:56


import os
import keras
from keras import layers
from keras import models


def cnnModel():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(26, 26, 32)))

    return model


if __name__ == "__main__":
    print(cnnModel().summary())
    # formula: (kernel_w*kernel_h*input_depth + 1) * output_depth
    # kernel_w*kernel_h*input_depth: 表示权重(weight)
    # 1: 表示偏置(bias)
    # output_depth: (feature map)滤波器个数
    # conv2d_1 (Conv2D)  param:160
    # (3*3*1 + 1)*16
    # conv2d_2 (Conv2D)  param:4640




