#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File film_scoring.py
# @ Description IMDB 电影评分
# @ Author alexchung
# @ Time 10/9/2019 AM 10:16


import os
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import models, layers
from keras import optimizers, losses, metrics
from keras.models import load_model


def loadImdbData(num_words=10000):
    """
    load data
    :param num_words: top 10000 most frequently occurring words in the training data
    :return:
    """
    return imdb.load_data(num_words=num_words)


def vectorizeSequences(sequences, dimension=10000):
    """
    向量化数据 one-hot
    :param sqquences: 数据序列
    :param dimension: 单词维度
    :return:
    """
    one_hot_results = np.zeros(shape=(len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        one_hot_results[i, sequence] = 1.
    return one_hot_results


def plotTrainValidationLossAccuracy(history):
    """
    Training validation loss and accuracy of epoch
    :param history: training parameter
    :return:
    """
    # history.keys() = ['val_loss', 'val_binary_accuracy', 'loss', 'binary_accuracy']
    history_dict = history.history
    loss_values = history_dict['loss']
    binary_accuracy = history_dict['binary_accuracy']
    val_binary_accuracy = history_dict['val_binary_accuracy']
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


def saveModel(model, model_path):
    """
    save model
    :param model: model
    :param model_name: model file name
    :return:
    """
    fold_path = os.getcwd() + '\\model'
    model_path = model_path
    try:
        if not os.path.exists(fold_path):
            os.mkdir(fold_path)
        # save model
        model.save(model_path)
    except:
        raise Exception('model save fail')


if __name__ == "__main__":
    (train_data, train_labels), (test_data, test_labels) = loadImdbData()

    # format transform
    x_train = vectorizeSequences(train_data)
    x_test = vectorizeSequences(test_data)

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    # split train_data, set apart 10000 samples as validation
    x_validation = x_train[:10000]
    x_training = x_train[10000:]

    y_validation = y_train[:10000]
    y_training = y_train[10000:]

    # model definition
    model = models.Sequential()
    # full connect layer(dense layer)
    model.add(layers.Dense(units=16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(units=16, activation='relu', input_shape=(16,)))
    model.add(layers.Dense(units=1, activation='sigmoid', input_shape=(16,)))

    # compile the model
    # rmspop； Root Mean Square Prop

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.binary_crossentropy,
                  metrics=[metrics.binary_accuracy])

    # training model
    history = model.fit(x=x_training, y=y_training, epochs=20, batch_size=512,
                        validation_data=(x_validation, y_validation))




    model_path = os.getcwd() + '\\model\\imdb.h5'
    # save model
    saveModel(model, model_path)

    # load model
    trained_model = load_model(model_path)

    # 评估测试集
    evaluate_result = trained_model.evaluate(x_test, y_test)
    print(evaluate_result)

    # 预测数据
    predict_prob = trained_model.predict(x_test)
    predict_label = np.zeros((predict_prob.shape,))

    for i, prop in enumerate(predict_prob):
        if prop < 0.5:
            predict_label[i] = 0
        else:
            predict_label[i] = 1
    print(predict_label)

    # plot train validation loss and accuracy
    plotTrainValidationLossAccuracy(history)

