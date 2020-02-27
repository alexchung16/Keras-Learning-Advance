#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : one_hot.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/2/27 上午9:57
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import string
from keras.preprocessing.text import Tokenizer

if __name__ == "__main__":
    samples = ["The cat sat on the mat.", "The dog ate my homework"]


    # word level
    token_index = {}
    for sample in samples:
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1

    max_length = 10

    results = np.zeros(shape=(len(samples), max_length, max(token_index.values())+1))

    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1.

    print(token_index)
    print(results)


    # character level
    characters = string.printable
    token_index = dict(zip(characters, range(1, len(characters)+1)))

    max_length = 50
    results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))

    for i, sample in enumerate(samples):
        for j, character in enumerate(sample):
            index = token_index.get(character)
            results[i, j, index] = 1.
    print(token_index)
    print(results)


    # word level use keras
    max_length = 100
    tokenizer = Tokenizer(num_words=max_length)
    tokenizer.fit_on_texts(samples)

    token_index = tokenizer.word_index

    sequence_result = tokenizer.texts_to_sequences(samples)
    one_hot_result = tokenizer.texts_to_matrix(samples, mode='binary')

    print(token_index)
    print(sequence_result)
    print(one_hot_result)

    # word level using hashing trick
    dimensionality = 1000
    max_length = 10

    results = np.zeros(shape=(len(samples), max_length, dimensionality))
    token_index = {}
    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:max_length]:
            index = abs(hash(word)) % dimensionality
            token_index [word] = index
            results[i, j, index] = 1.

    print(token_index)
    print(results)















