#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File visualize_conv.py
# @ Description
# @ Author alexchung
# @ Time 1/10/2019 PM 14:27


import os
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers
from keras import optimizers, losses
from keras.preprocessing import image

# model path
model_path = os.path.join(os.getcwd(), 'model')
# train data path
data_path = os.path.join(os.getcwd(), 'data')
# image path
image_path = os.path.join(os.getcwd(), 'image')

# origin dataset
original_dataset_dir = '/home/alex/Documents/datasets/dogs-vs-cats/train'
# separate dataset
base_dir = '/home/alex/Documents/datasets/dogs_and_cat_separate'

# train dataset
train_dir = os.path.join(base_dir, 'train')
# validation dataset
val_dir = os.path.join(base_dir, 'validation')
# test dataset
test_dir = os.path.join(base_dir, 'test')

# train cat dataset
train_cat_dir = os.path.join(train_dir, 'cat')
# train dog dataset
train_dog_dir = os.path.join(train_dir, 'dog')

# validation cat dataset
val_cat_dir = os.path.join(val_dir, 'cat')
# validation cat dataset
val_dog_dir = os.path.join(val_dir, 'dog')

# test cat dataset
test_cat_dir = os.path.join(test_dir, 'cat')
test_dog_dir = os.path.join(test_dir, 'dog')


img_path = original_dataset_dir + '/cat.1700.jpg'


def visualizeAcitivitionLayer(model, layer_num, img_tensor):
    """
    visualization activation layer
    :param model:
    :param layer_num:
    :param image:
    :return:
    """
    # model instant
    layer_outputs = [layer.output for layer in model.layers[:layer_num]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activation = activation_model.predict(img_tensor)
    layer_names = []
    # layer_names = [[layer.name for layer in model.layers[i]] for  i in layer_lists]
    layer_names = [layer.name for layer in model.layers[:layer_num]]

    # the row num of image show
    n_row = 16

    for layer_name, layer_activate in zip(layer_names, activation):

        # feature num of the layer
        n_features = layer_activate.shape[-1]
        # feature image size
        length_feature_size = layer_activate.shape[1]
        width_feature_size = layer_activate.shape[2]

        n_col = n_features // n_row
        # flat image
        display_grid = np.zeros((length_feature_size*n_col, width_feature_size*n_row))

        for col in range(n_col):
            for row in range(n_row):
                channel_image = layer_activate[0, :, :, col*n_row+row]

                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                # adjustment pixel size range to between in 0 and 255
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                display_grid[col*width_feature_size: (col+1)*width_feature_size,
                             row*length_feature_size: (row+1)*length_feature_size] = channel_image

        # 尺度转换
        weigth_scale = 1. / length_feature_size
        height_scale = 1. / width_feature_size

        plt.figure(figsize=(weigth_scale*display_grid.shape[1], height_scale*display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        if os.path.exists(image_path):
            pass
        else:
            os.mkdir(image_path)
        plt.savefig(image_path + '/{0}.jpg'.format(layer_name))
        # plt.show()



if __name__ == "__main__":
    img = image.load_img(path=img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = img_tensor/255.

    model = models.load_model(model_path+'/cnn_net.h5')

    visualizeAcitivitionLayer(model, 6, img_tensor)