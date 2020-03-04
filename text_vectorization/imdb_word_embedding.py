#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : imdb_word_embedding.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/3/4 下午3:17
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense


imdb_dir = '/home/alex/Documents/dataset/aclImdb'
glove_dir = '/home/alex/Documents/pretraing_model/glove.6B'


def load_imdb(imdb_dir):
    """

    :param dataset_dir:
    :param type:
    :return:
    """
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []
    for type in ['train', 'test']:
        data_dir = os.path.join(imdb_dir, type)
        for label_index, label_type in enumerate(['neg', 'pos']):
            label_dir = os.path.join(data_dir, label_type)
            for index, text_file in enumerate(os.listdir(label_dir)):
                if text_file.split('.')[-1] == 'txt':
                    with open(os.path.join(label_dir, text_file), 'r') as f:
                        if type == 'train':
                            train_texts.append(f.read())
                            train_labels.append(label_index)
                        elif type == 'test':
                            test_texts.append(f.read())
                            test_labels.append(label_index)
    return (train_texts, train_labels), (test_texts, test_labels)



if __name__ == "__main__":


    # ------------------------------step 1 dataset preprocessing---------------------------
    # load imdb
    (texts_train, labels_train), (texts_test, labels_test) = load_imdb(imdb_dir=imdb_dir)

    # tokenize word to sequence
    max_len = 100  # cut reviews after 100 words
    train_samples = 200
    val_samples = 10000
    max_words = 10000  # considers top 10000 words in the dataset

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts_train)
    sequence = tokenizer.texts_to_sequences(texts_train)
    # print(len(sequence)) # 25000

    # get word index
    word_index = tokenizer.word_index
    # index_word = tokenizer.index_word
    # print(len(word_index)) # 88582

    # sequence padding (samples, max)->(25000, 100)
    train_sequence = pad_sequences(sequence, maxlen=max_len)
    train_labels =  np.asarray(labels_train)
    # print(train_sequence.shape) # (25000, 100)
    # print(train_labels.shape) # (25000,)

    # shuffle dataset
    indices = np.arange(train_sequence.shape[0])
    np.random.shuffle(indices)
    train_sequence = train_sequence[indices]
    train_labels = train_labels[indices]

    # split dataset
    x_train = train_sequence[: train_samples]
    y_train = train_labels[: train_samples]
    x_val = train_sequence[train_samples: train_samples+val_samples]
    y_val = train_labels[train_samples: train_samples + val_samples]

    # ------------------------------step 2 parse GloVe(Global Vectors) model---------------------------
    embedding_index = {}
    # use 100 dimension embedding vector
    # embedding_dim = max_len
    glove_model = os.path.join(glove_dir, 'glove.6B.100d.txt')
    with open(glove_model, 'r') as f:
        for line in f:
            # print(len(line.split())) #(1 + 100)
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype=np.float32)
            embedding_index[word] = vector


    # prepare GloVe matrix
    embedding_dim = 100
    #
    embedding_matrix = np.zeros(shape=(max_words, embedding_dim))
    for word, index in word_index.items():
        if index < max_words:
            # get corresponding vector of words
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    # ------------------------------step 3 construct network and load glove weight---------------------------
    # network
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.summary()

    # load glove to network and frozen layer
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False


    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(x=x_train, y=y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=[x_val, y_val])


















