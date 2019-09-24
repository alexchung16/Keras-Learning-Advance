#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File mnist.py
# @ Description
# @ Author alexchung
# @ Time 24/9/2019 AM 10:19

import os
import keras
from keras import layers
from keras import models
from keras import optimizers,losses, metrics
from keras.datasets import mnist
from keras.utils import to_categorical


tb_path = os.getcwd() + r'/logs'

def loadMNISTData():
    """
    load data
    :return:
    """
    return mnist.load_data()

# MNIST model
def MNIST():
    """
    MNIST model
    :return:
    """
    model = models.Sequential()
    # CNN
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # 3D 平展为1D
    model.add(layers.Flatten())
    # FCN 分类
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model


if __name__ == "__main__":
    tb_cb = keras.callbacks.TensorBoard(log_dir=tb_path, histogram_freq=1, write_images=1)

    model = MNIST()
    (train_images, train_labels), (test_images, test_labels) = loadMNISTData()

    # transform data
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    print(train_images.shape)

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model.compile(optimizer=optimizers.rmsprop(lr=0.001),
                  loss=losses.categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(train_images[10000:], train_labels[10000:], epochs=5, batch_size=64, callbacks=[tb_cb],
              validation_data=(train_images[:10000], train_labels[:10000]))

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(test_acc)




