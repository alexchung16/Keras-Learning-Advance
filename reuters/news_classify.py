#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File news_classify.py
# @ Description reuters 新闻分类
# @ Author alexchung
# @ Time 10/9/2019 PM 16:34

import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import reuters
from keras import models, layers
from keras import optimizers, losses, metrics
from keras.models import load_model

# model path
model_path = os.path.join(os.getcwd(), 'model')
# tensorboard path
tb_path = os.path.join(os.getcwd() + 'logs')


def loadReutersData(num_words=10000):
    """
    load data
    :param num_words: top 10000 most frequently occurring words in the training data
    :return:
    """
    return reuters.load_data(num_words=num_words)


def vectorizeSequences(sequences, dimension=10000):
    """
    向量化数据 one-hot
    :param sqquences: 数据序列
    :param dimension: 单词最大维度
    :return:
    """
    vector_results = np.zeros(shape=(len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        vector_results[i, sequence] = 1.
    return vector_results


def oneHotEncode(labels, dimension=46):
    """
    label one hot encode
    :param label: label
    :param dimension: 标签最大维度
    :return:
    """
    one_hot_results = np.zeros(shape=(len(labels), dimension))
    for i, label in enumerate(labels):
        one_hot_results[i, label] = 1
    return one_hot_results


def plotTrainValidationLossAccuracy(history):
    """
    Training validation loss and accuracy of epoch
    :param history: training parameter
    :return:
    """
    # history.keys() = ['val_loss', 'val_categorical_accuracy', 'loss', 'categorical_accuracy']
    history_dict = history.history
    loss_values = history_dict['loss']
    binary_accuracy = history_dict['categorical_accuracy']
    val_binary_accuracy = history_dict['val_categorical_accuracy']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.plot(epochs, loss_values, 'bo', label='Training loss')
    ax1.plot(epochs, val_loss_values, 'b', label='Validation loss')
    ax1.set_title('Training and validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, binary_accuracy, 'bo', label='Training accuracy')
    ax2.plot(epochs, val_binary_accuracy, 'b', label='Validation accuracy')
    ax2.set_title('Training and validation accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.show()


def saveModel(model, model_name):
    """
    save model
    :param model: model
    :param model_name: model file name
    :return:
    """
    try:
        if os.path.exists(model_path) is False:
            os.mkdir(model_path)
        # save model
        model.save(os.path.join(model_path, model_name))
    except:
        raise Exception('model save failed')


if __name__ == "__main__":

    tb_cb = keras.callbacks.TensorBoard(log_dir=tb_path, histogram_freq=1, write_images=1)
    (train_data, train_labels), (test_data, test_labels) = loadReutersData()
    # one-hot 向量化单词序列
    x_train = vectorizeSequences(train_data)
    x_test = vectorizeSequences(test_data)

    y_train = oneHotEncode(train_labels)
    y_test = oneHotEncode(test_labels)

    # split train_data, set apart 1000 samples as validation
    x_validation = x_train[:1000]
    x_training = x_train[1000:]

    y_validation = y_train[:1000]
    y_training = y_train[1000:]

    # model definition
    model = models.Sequential()
    model.add(layers.Dense(units=64, activation='relu', input_shape=(10000, )))
    model.add(layers.Dense(units=64, activation='relu', input_shape=(64, )))
    model.add(layers.Dense(units=46, activation='softmax', input_shape=(64, )))

    # compile model
    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.categorical_crossentropy,
                  metrics=[metrics.categorical_accuracy]
                  )

    # train model
    history = model.fit(x=x_training, y=y_training, epochs=20, batch_size=512, callbacks=[tb_cb],
                        validation_data=(x_validation, y_validation))

    # plot train validation loss and accuracy
    plotTrainValidationLossAccuracy(history)

    # 创建模型保存路径
    # save model
    saveModel(model, 'reuters.h5')

    # load model
    trained_model = load_model(os.path.join(model_path, 'reuters.h5'))

    # evaluate test
    evaluate_result = trained_model.evaluate(x_test, y_test)
    print(evaluate_result)

    # 预测数据
    predict_prob = trained_model.predict(x_test)
    predict_label = np.zeros((predict_prob.shape[0],))
    for i, probs in enumerate(predict_prob):
        predict_label[i] = np.argmax(probs)
        # probs_hashmap = {}
        # for j, prob in enumerate(probs):
        #     probs_hashmap[prob] = j
        # predict_label[i] = probs_hashmap[max(probs)]
    print(predict_label)







