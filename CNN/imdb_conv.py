#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : imdb_conv.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/3/6 下午5:43
# @ Software   : PyCharm
#-------------------------------------------------------

import os
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, Dense, Conv1D, MaxPool1D, GlobalMaxPool1D
from keras.optimizers import RMSprop


if __name__ == "__main__":


    max_words = 10000
    max_len = 500  # cut off after 500 words
    output_dim = 32

    # data preprocess
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

    input_train = sequence.pad_sequences(x_train, maxlen=max_len)
    input_test = sequence.pad_sequences(x_test, maxlen=max_len)

    # construct network
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=output_dim, input_length=max_len))
    model.add(Conv1D(filters=32, kernel_size=7, activation='relu'))
    model.add(MaxPool1D(pool_size=5))
    model.add(Conv1D(filters=16, kernel_size=7, activation='relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dense(units=1))
    model.summary()

    model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x=input_train, y=y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)
