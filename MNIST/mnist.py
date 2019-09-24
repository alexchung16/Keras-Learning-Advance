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
from keras import optimizers, losses, metrics
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt


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


def plotTrainValidationLossAccuracy(history):
    """
    Training validation loss and accuracy of epoch
    :param history: training parameter
    :return:
    """
    # history.keys() = ['val_loss', 'val_binary_accuracy', 'loss', 'binary_accuracy']
    history_dict = history.history
    train_loss = history_dict['loss']
    train_accuracy = history_dict['acc']
    val_accuracy = history_dict['val_acc']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(train_loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.plot(epochs, train_loss, 'bo', label='Training loss')
    ax1.plot(epochs, val_loss, 'b', label='Validation loss')
    ax1.set_title('Training and validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, train_accuracy, 'bo', label='Training accuracy')
    ax2.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    ax2.set_title('Training and validation accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.show()


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
    history = model.fit(train_images[10000:], train_labels[10000:], epochs=5, batch_size=64, callbacks=[tb_cb],
              validation_data=(train_images[:10000], train_labels[:10000]))

    print(history.history.keys())
    plotTrainValidationLossAccuracy(history)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(test_acc)




