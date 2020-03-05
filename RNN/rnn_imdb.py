#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : rnn_imdb.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/3/5 上午10:16
# @ Software   : PyCharm
#-------------------------------------------------------

import os
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, Dense, SimpleRNN

if __name__ == "__main__":

    max_feature = 10000
    max_len = 500 # cut off after 500 words
    output_dim = 32

    # data preprocess
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_feature)

    input_train = sequence.pad_sequences(x_train, maxlen=max_len)
    input_test = sequence.pad_sequences(x_test, maxlen=max_len)
    # print(input_train.shape) # (25000, 500)
    # print(input_test.shape) # (25000, 500)

    # RNN network
    model = Sequential()
    model.add(Embedding(input_dim=max_feature, output_dim=output_dim, input_length=max_len))
    model.add(SimpleRNN(units=32, return_sequences=True))
    model.add(SimpleRNN(units=16))
    model.add(Dense(units=1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x=input_train, y=y_train,
                        epochs=10,
                        batch_size=128,
                        validation_split=0.2)

