#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File multiple_input_single_output.py
# @ Description
# @ Author alexchung
# @ Time 6/10/2019 PM 15:42

import os
import numpy as np
from keras.models import Model
from keras import layers
from keras import Input
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy


def multipleOutput(vocabulary_size, num_income_groups):
    """
    multiple output model
    :param vocabulary_size:
    :param num_income_groups:
    :return:
    """

    post_input = layers.Input(shape=(None, ), dtype='int32', name='posts')
    embedded_post = layers.Embedding(input_dim=256, output_dim=vocabulary_size)(post_input)
    # Conv1D layer
    x = layers.Conv1D(filters=128, kernel_size=5, activation='relu')(embedded_post)
    x = layers.MaxPooling1D(pool_size=5)(x)
    x = layers.Conv1D(filters=256, kernel_size=5, activation='relu')(x)
    x = layers.Conv1D(filters=256, kernel_size=5, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=5)(x)
    x = layers.Conv1D(filters=256, kernel_size=5, activation='relu')(x)
    x = layers.Conv1D(filters=256, kernel_size=5, activation='relu')(x)
    x = layers.GlobalMaxPool1D()(x)

    # Dense layer
    # regression problem
    age_prediction = layers.Dense(units=1, name='age')(x)
    # category category problem
    income_prediction = layers.Dense(units=num_income_groups, activation='softmax', name='income')(x)
    # binary category
    gender_prediction = layers.Dense(units=1, activation='sigmoid', name='gender')(x)

    # model
    model = Model(inputs=post_input, outputs=[age_prediction, income_prediction, gender_prediction])

    model.compile(optimizer=RMSprop(lr=1e-5),
                  loss={'age': 'mse', 'income': 'categorical_crossentropy', 'gender': 'binary_crossentropy'},
                  loss_weights={'age': 0.25, 'income': 1., 'gender': 10.}
                  )
    return model


if __name__ == "__main__":
    vocabulary_size = 50000
    num_income_groups = 10
    model = multipleOutput(vocabulary_size, num_income_groups)
    print(model.summary())

    # model.fit(posts, [age_targets, income_targets, gender_targets], epochs=10, batch_size=64)
    # model.fit(posts, {'age': age_targets,
    #                   'income': income_targets,
    #                   'gender': gender_targets},
    #           epochs=10, batch_size=64)