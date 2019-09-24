#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File housing_predict.py
# @ Description housing 房价预测
# @ Author alexchung
# @ Time 12/9/2019 AM 09:27


import os
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
from keras import models, layers
from keras import optimizers, losses, metrics
from keras.models import load_model

# tensorboard path
tb_path = os.getcwd()

def loadHousingData():
    """
    load data
    :return:
    """
    return boston_housing.load_data()


def dataStandardize(train_data:np.array, test_data:np.array):
    """
    data standardization
    :param data:
    :return:
    """
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std
    test_data  = (test_data - mean) / std
    return train_data, test_data


def buildMode():
    """
    build mode
    :return:
    """
    model = models.Sequential()
    model.add(layers.Dense(units=64, activation='relu', input_shape=(13,)))
    model.add(layers.Dense(units=64, activation='relu',  input_shape=(64,)))
    model.add(layers.Dense(units=1, input_shape=(64,)))

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss='mse',
                  metrics=['mae']
                  )
    return model


if __name__ == "__main__":
    (train_data, train_targets), (test_data, test_labels) = loadHousingData()
    # data standardization
    train_data, test_data = dataStandardize(train_data, test_data)

    model = buildMode()

    k = 5
    num_epochs = 100
    all_mae_histories = []
    num_val_samples = len(train_data) // k

    for i in range(k):
        print('processing fold #', i)
        # validation data
        validate_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
        validate_targets = train_targets[i*num_val_samples:(i + 1) * num_val_samples]
        # training data
        training_data = np.concatenate([train_data[:i * num_val_samples],
                                        train_data[(i + 1) * num_val_samples:]], axis=0)
        training_targets = np.concatenate([train_targets[:i * num_val_samples],
                                           train_targets[(i + 1) * num_val_samples:]], axis=0)

        model = buildMode()
        history = model.fit(x=training_data, y=training_targets, epochs=num_epochs,
                            validation_data=(validate_data, validate_targets), batch_size=1,  verbose=0)
        mae_history = history.history['val_mean_absolute_error']
        all_mae_histories.append(mae_history)

    # 获取平均MAE
    average_mae_history = [
        np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

    plt.plot(range(1, len(average_mae_history)+1), average_mae_history)
    plt.xlabel('Epoch')
    plt.ylabel('Validation MAE')
    plt.show()






