#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : rnn_demo.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/3/5 上午9:45
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np

from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN


# loop/state

time_step = 100
input_feature  = 32
output_feature = 64

if __name__ == "__main__":


    # --------------------------------------part 1 rnn demo----------------------------------
    # initial input and state
    inputs = np.random.random(size=(time_step, input_feature))
    state_t = np.zeros(shape=(output_feature, ))

    np.random.seed(0)
    # create random weight and bias
    W = np.random.random((output_feature, input_feature)) # feedforward weight
    U = np.random.random((output_feature, output_feature)) # rnn update weight
    b = np.random.random((output_feature, )) # bias

    successive_outputs = []

    for input_t in inputs:
        output_t = np.tanh(np.dot(W, input_t.T) + np.dot(U, state_t) + b)
        successive_outputs.append(output_t)
        # update state
        state_t = output_t

    sequence_outputs = np.concatenate(successive_outputs, axis=0)
    print(sequence_outputs)

    # --------------------------------------part 2 rnn by keras----------------------------------
    model = Sequential()

    model.add(Embedding(input_dim=10000, output_dim=32, input_length=20))
    # return_sequence: True -> return shape(batch_size, time_steps, output_feature)
    #                  False -> return shape(batch_size, output_feature) only the output at last time_step
    # output shape -> (None, 20, 32)
    model.add(SimpleRNN(units=32, return_sequences=True))
    # output shape -> (None, 20, 32)
    model.add(SimpleRNN(units=32, return_sequences=True))
    # output shape -> (None, 32)
    model.add(SimpleRNN(units=32))
    model.summary()






