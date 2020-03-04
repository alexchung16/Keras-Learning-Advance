#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : word_embedding.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/3/4 上午9:48
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
from keras.datasets import imdb
from keras import preprocessing

from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

def convert_index_to_word(index_list, index_dict):
    """
    index list
    index dict
    :param index:
    :param index_dict:
    :return:
    """
    return ' '.join([index_dict.get(index, '?') for  index in index_list])


if __name__ == "__main__":


    max_featurte = 10000
    max_len = 20

    # step 1 imdb dataset preprocessing
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_featurte)
    # print(len(x_train)) # 2500

    # imdb_index process
    imdb_index = imdb.get_word_index()
    # print(len(imdb_index)) # 88584
    imdb_index = dict([(word, index + 3) for word, index in imdb_index.items()])
    imdb_index['<PAD>'] = 0 # for extend sentence to equal size of length
    imdb_index['STARD'] = 1 # for start
    imdb_index['UNK'] = 2 # for UNKNOWN
    imdb_index['UNSET'] = 3
    imdb_dict = dict([(index, word) for word, index in imdb_index.items()])
    # print(convert_index_to_word(index_dict=imdb_dict, index_list=x_train[0]))

    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len, padding='post', value=imdb_index['<PAD>'])
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len, padding='post', value=imdb_index['<PAD>'])
    print(x_train.shape) # (25000, 20)

    # step 2 construct network
    vocab_size = 10000
    model = Sequential()
    # (samples, max_len)->(25000, 20)
    # Embedding 理解为字典，将证书索引，映射为密集向量
    model.add(Embedding(input_dim=vocab_size, output_dim=8, input_length=max_len))
    #(samples, max_len, output_dim)->(25000, 20, 8)
    model.add(Flatten())
    # (samples, max_len*output_dim)->(2500, 160)
    model.add(Dense(1, activation='sigmoid'))
    # (samples,1) ->(25000, 1)
    model.summary()

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

    history = model.fit(x=x_train, y=y_train,
                        epochs=10,
                        batch_size=32,
                        validation_split=0.2)

















    #