#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File mnist.py
# @ Description
# @ Author alexchung
# @ Time 24/9/2019 AM 10:19

import os
import keras
from keras import layers, models
from keras import optimizers, losses, metrics

from keras import Model
from keras.layers import Input, Flatten, Dense
# from keras.layers.convolutional import Conv2D, MaxPooling2D
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
    # Conv2D parameter illustrate
    # filters 滤波器个数或特征图深度
    # kernel_size 卷积核尺寸
    # activation 激活函数
    # input_shape 输入图像尺寸
    # strides 步幅(步进卷积)， 默认(1,1)
    # padding 是否填充
    # stride
    # input shape (28, 28, 1) output shape(26, 26, 1)
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1),
                            strides=(1, 1), padding='valid'))
    # input shape (26, 26, 32) output shape(13, 13, 32)
    model.add(layers.MaxPool2D(2, 2))
    # input shape (13, 13, 32) output shape(11, 11, 64)
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(13, 13, 32)))
    # input shape (11, 11, 64) output shape(5, 5, 64)
    model.add(layers.MaxPool2D(2, 2))
    # input shape (5, 5, 64) output shape(3, 3, 128)
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(5, 5, 64)))
    # 3D 平展为1D
    # input shape (3, 3, 64) output shape(3*3*64=1152, )
    model.add(layers.Flatten())
    # FCN 分类
    # input shape (1152, ) output shape(64, )
    model.add(layers.Dense(64, activation='relu'))
    # input shape (64, ) output shape(10, )
    model.add(layers.Dense(10, activation='softmax'))

    return model


def MNISTFunctionAPI():
    """
    keras function API realization
    :return:
    """
    input_tensor = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(input_tensor)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2 ,2))(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=64, activation='relu')(x)
    output_tensor = layers.Dense(units=10, activation='softmax')(x)
    model = Model(inputs= input_tensor, outputs=output_tensor)
    return model


def plotTrainValidationLossAccuracy(history):
    """
    Training validation loss and accuracy of epoch
    :param history: training parameter
    :return:
    """
    # history.keys() = ['val_loss', 'val_binary_accuracy', 'loss', 'binary_accuracy']
    history_dict = history.history
    loss = history_dict['loss']
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.plot(epochs, loss, 'bo', label='Training loss')
    ax1.plot(epochs, val_loss, 'b', label='Validation loss')
    ax1.set_title('Training and validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, acc, 'bo', label='Training accuracy')
    ax2.plot(epochs, val_acc, 'b', label='Validation accuracy')
    ax2.set_title('Training and validation accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.show()


if __name__ == "__main__":
    tb_cb = keras.callbacks.TensorBoard(log_dir=tb_path, histogram_freq=1, write_images=1)

    # seq_model = MNIST()
    # print(seq_model.summary())
    model = MNISTFunctionAPI()
    print(model.summary())
    (train_images, train_labels), (test_images, test_labels) = loadMNISTData()

    # transform data
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255


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





