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
from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing

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


    max_featurte = 1000
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

    # print(imdb_dict)
    # print(convert_index_to_word(index_dict=imdb_dict, index_list=x_train[0]))
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len, padding='post', value=imdb_index['<PAD>'])
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len, padding='post', value=imdb_index['<PAD>'])









    #