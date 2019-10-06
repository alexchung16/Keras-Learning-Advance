#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File multiple_input.py
# @ Description
# @ Author alexchung
# @ Time 6/10/2019 PM 14:42

import os
import numpy as np
from keras.models import Model
from keras import layers
from keras import Input
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy


def multipleInput(text_vocabulary_size, question_vocabulary_size, answer_vocabulary_size):

    text_input = Input(shape=(None, ), dtype='int32', name='text')
    embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
    encode_text = layers.LSTM(units=32)(embedded_text)

    question_input = Input(shape=(None,), dtype='int32', name='question')
    embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
    encode_question = layers.LSTM(units=16)(embedded_question)

    # concatenate multiple input layer
    concatenate =  layers.concatenate(inputs=[encode_text, encode_question], axis=1)

    answer  = layers.Dense(units=answer_vocabulary_size, activation='softmax')(concatenate)

    model = Model(inputs=[text_input, question_input], outputs=answer)

    model.compile(optimizer=RMSprop(lr=1e-5),
                  loss=categorical_crossentropy,
                  metrics=['acc'])
    return model


if __name__ == "__main__":
    text_vocabulary_size = 10000
    question_vocabulary_size = 10000
    answer_vocabulary_size = 500
    model = multipleInput(text_vocabulary_size, question_vocabulary_size, answer_vocabulary_size)
    print(model.summary())

    # model.fit([text, question], answers, epochs=10, batch_size=128)
    # model.fit({'text': text, 'question': question}, answers,epochs=10, batch_size=128)