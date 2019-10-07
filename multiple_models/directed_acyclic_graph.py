#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File directed_acyclic_graph.py
# @ Description
# @ Author alexchung
# @ Time 6/10/2019 PM 19:22

import os
import numpy as np
from keras.models import Model
from keras import layers
from keras import Input
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy


def inceptionModule():
    """
    inception module
    :return:
    """
    input_tensor = layers.Input(shape=(224, 224, 3), name='inception')
    x_a = layers.Conv2D(filters=128, kernel_size=1, activation='relu', strides=2)(input_tensor)

    x_b = layers.Conv2D(filters=128, kernel_size=1, activation='relu')(input_tensor)
    x_b = layers.Conv2D(filters=128, kernel_size=2, activation='relu', strides=2)(x_b)

    x_c = layers.AveragePooling2D(pool_size=2, strides=2)(input_tensor)
    x_c = layers.Conv2D(filters=128, kernel_size=1, activation='relu')(x_c)

    x_d = layers.Conv2D(filters=128, kernel_size=1, activation='relu')(input_tensor)
    x_d = layers.Conv2D(filters=128, kernel_size=1, activation='relu')(x_d)
    x_d = layers.Conv2D(filters=128, kernel_size=2, activation='relu', strides=2)(x_d)

    output_tensor = layers.concatenate(inputs=[x_a, x_b, x_c, x_d], axis=-1)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model


def identityResidualConnect():
    """
    identity residual connect when the feature map size are same
    :return:
    """

    x = layers.Input(shape=(224, 224, 128))
    y = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    y = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(y)
    y = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(y)

    y = layers.add([y, x])

    model = Model(inputs=x, outputs=y)

    return model


def linearResidualConnect():
    """
    linear residual connect when the feature map size are different
    :return:
    """

    x = layers.Input(shape=(224, 224, 3))
    y = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    y = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(y)
    y = layers.MaxPooling2D(pool_size=2, strides=2)(y)

    residual = layers.Conv2D(filters=128, kernel_size=1, strides=2, padding='same')(x)
    y = layers.add([y, residual])

    model = Model(inputs=x, outputs=y)

    return model

if __name__ == "__main__":
    # model = inceptionModule()
    # model = identityResidualConnect()
    model = linearResidualConnect()
    print(model.summary())