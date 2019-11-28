# -*- coding: utf-8 -*-
# @ File densenet_50.py
# @ Description
# @ Author alexchung
# @ Time 28/11/2019 PM 15:52


from keras.models import Model
from keras.layers import Input, merge, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K

from DenseNet.custom_layers import Scale

class DenseNet():
    def __init__(self, input_shape, num_classes=2, num_filters=64, growth_k=32, num_dense_block=4, num_layers=None,
                 reduction=0.0, drop_rate=0.0, weight_decay=1e-4, batch_norm_epsilon=1e-5):

        self.inputs = Input(shape=input_shape, name='data')
        self.num_classes=num_classes
        self.growth_k = growth_k
        self.num_dense_block =num_dense_block
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.compression =1.0 - reduction
        self.drop_rate=drop_rate
        self.weight_decay = weight_decay
        self.batch_norm_epsilon = batch_norm_epsilon
        self.concat_axis = 3

    def inference(self):
        predict = self.dense_net(inputs=self.inputs, num_classes=self.num_classes, num_filters=self.num_filters,
                               num_dense_block=self.num_dense_block, num_layers=self.num_layers,
                               grows_k=self.growth_k, drop_rate=self.drop_rate,
                               batch_norm_epsilon=self.batch_norm_epsilon, compression=self.compression)

        model = Model(inputs=self.inputs, outputs=predict)
        return model

    def dense_net(self, inputs, num_classes, num_filters, num_dense_block, num_layers, grows_k, drop_rate,
                  batch_norm_epsilon=1e-5, compression=1.0):
        # 224 x 224 x 3
        x = ZeroPadding2D(padding=(3, 3), name='conv1_zeropadding')(inputs)
        # 230 x 230 x 3
        x = Convolution2D(filters=num_filters, kernel_size=(7, 7), strides=(2, 2), use_bias=False,
                          name='conv1', padding='valid')(x)
        # 112 x 112 x 3
        x = Scale(axis=self.concat_axis, name='conv1_scale')(x)
        # 112 x 112 x 3
        x = ZeroPadding2D(padding=(1, 1), name='pool1_zeropadding')(x)
        # 114 x 114 x 3
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)
        # 56 x 56 x3

        # dense_block
        for block_id in range(num_dense_block - 1):
            stage = block_id + 2
            x, num_filters = self.dense_block(inputs=x, stage=stage, num_layers=num_layers[block_id],
                                              number_filters=num_filters, growth_k=grows_k, drop_rate=drop_rate,
                                              batch_norm_epsilon=batch_norm_epsilon)
            x = self.transition_block(inputs=x, stage=stage, num_filters=num_filters, compression=compression,
                                      drop_rate=drop_rate)
            num_filters = int(compression * num_filters)

        final_stage =  num_dense_block + 1
        x, num_filters = self.dense_block(inputs=x, stage=final_stage, num_layers=num_layers[-1],
                                          number_filters=num_filters, growth_k=grows_k, drop_rate=drop_rate,
                                          batch_norm_epsilon=batch_norm_epsilon)
        x = BatchNormalization(epsilon=batch_norm_epsilon, name='conv' + str(final_stage)+'_blk_bn')(x)
        x = Scale(axis=self.concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
        x = Activation(activation='relu', name='relu'+str(final_stage)+'_blk_scale')(x)
        x = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)

        x = Dense(units=num_classes, name='fc6')(x)
        predict = Activation(activation='softmax', name='predict')(x)

        return predict

    def bottleneck_block(self, inputs, stage, branch, num_filters, drop_rate=None, batch_norm_epsilon=1e-5):
        """

        :param inputs:
        :param stage:
        :param branch:
        :param num_filters:
        :param drop_rate:
        :param batch_norm_epsilon:
        :return:
        """
        conv_name_base = 'conv' + str(stage) + '_' + str(branch)
        relu_name_base = 'relu' + str(stage) + '_' + str(branch)

        # bottleneck layers
        inter_channel = num_filters * 4
        x = BatchNormalization(axis=self.concat_axis, epsilon=batch_norm_epsilon, name=conv_name_base+'_x1_bn')(inputs)
        x = Scale(axis=self.concat_axis, name=conv_name_base+'_x1_scale')(x)
        x = Activation(activation='relu', name=relu_name_base+'x1')(x)
        x = Convolution2D(filters=inter_channel, kernel_size=(1, 1), strides=(1, 1), use_bias=False,
                          name=conv_name_base+'_x1', padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = BatchNormalization(axis=self.concat_axis, epsilon=batch_norm_epsilon, name=conv_name_base+'_x2_bn')(x)
        x = Scale(axis=self.concat_axis, name=conv_name_base + '_x2_scale')(x)
        x = Activation(activation='relu', name=relu_name_base + 'x2')(x)
        x = ZeroPadding2D(padding=(1, 1), name=conv_name_base+'_x2_zeropadding')(x)
        x = Convolution2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), use_bias=False,
                          name=conv_name_base+'_2', padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        return x

    def dense_block(self, inputs, stage, num_layers, number_filters, growth_k, drop_rate=None, batch_norm_epsilon=1e-5,
                    growth_filters=True):
        """

        :param inputs:
        :param state:
        :param num_layers:
        :param num_filters:
        :param growth_k:
        :param drop_rate:
        :param weight_decay:
        :param growth_filters:
        :return:
        """
        concat_feat = inputs

        for n in range(num_layers):
            branch = n + 1
            x = self.bottleneck_block(inputs=concat_feat, stage=stage, branch=branch, num_filters=growth_k,
                                      drop_rate=drop_rate, batch_norm_epsilon=batch_norm_epsilon)

            concat_feat = merge.concatenate(inputs=[concat_feat, x], axis=self.concat_axis,
                                            name='concat_'+str(stage)+'_'+str(branch))
            if growth_filters:
                number_filters += growth_filters
        return concat_feat, number_filters

    def transition_block(self, inputs, stage, num_filters, compression=1.0, drop_rate=None, batch_norm_epsilon=1e-5):
        """

        :param inputs:
        :param stage:
        :param num_filters:
        :param compression:
        :param drop_rate:
        :return:
        """
        conv_name_base = 'conv' + str(stage) + '_blk'
        relu_name_base = 'relu' + str(stage) + '_blk'
        pool_name_base = 'pool' + str(stage) + '_blk'

        x = BatchNormalization(epsilon=batch_norm_epsilon, axis=self.concat_axis, name=conv_name_base+'_bn')(inputs)
        x = Scale(axis=self.concat_axis, name=conv_name_base+'_scale')(x)
        x = Activation('relu', name=relu_name_base)(x)
        x = Dropout(rate=drop_rate)(x)

        x = Convolution2D(filters=int(num_filters*compression), kernel_size=(1, 1), strides=(1, 1), use_bias=False,
                          name=conv_name_base, padding='valid')(x)
        x = Dropout(rate=drop_rate)(x)

        x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name=pool_name_base)(x)

        return x












