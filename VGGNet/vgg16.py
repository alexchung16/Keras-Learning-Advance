#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File vgg16.py
# @ Description
# @ Author alexchung
# @ Time 25/9/2019 AM 09:55

import os
import shutil
import keras
from keras import layers
from keras import models
from keras import optimizers, losses, metrics
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

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


def makedir(path):
    """
    create dir
    :param path:
    :return:
    """
    if os.path.exists(path) is False:
        os.mkdir(path)


try:
    if os.path.exists(original_dataset_dir) is False:
        print('dataset is not exist please check the path')
    else:
        if os.path.exists(base_dir) is False:
            os.mkdir(base_dir)
            print('{0} has been created'.format(base_dir))
        else:
            print('{0} has been exist'.format(base_dir))

        makedir(train_dir)
        makedir(val_dir)
        makedir(test_dir)

        makedir(train_cat_dir)
        makedir(train_dog_dir)
        makedir(val_cat_dir)
        makedir(val_dog_dir)
        makedir(test_cat_dir)
        makedir(test_dog_dir)

except FileNotFoundError as e:
    print(e)


def structureDataset(train_num=5000, val_num=1000, test_num=1000):
    """
    structure train dataset validation dataset test dataset by separate origin dataset
    :param train_num: train dataset num
    :param val_num: validation dataset num
    :param test_num: test dataset num
    :return:
    """
    # train dataset image name list
    train_cat_list = ['cat.{0}.jpg'.format(i) for i in range(0, train_num)]
    train_dog_list = ['dog.{0}.jpg'.format(i) for i in range(0, train_num)]

    # validation dataset image name list
    val_cat_list = ['cat.{0}.jpg'.format(i) for i in range(train_num, train_num+val_num)]
    val_dog_list = ['dog.{0}.jpg'.format(i) for i in range(train_num, train_num+val_num)]

    # test dataset image name list
    test_cat_list = ['cat.{0}.jpg'.format(i) for i in range(train_num+val_num, train_num+val_num+test_num)]
    test_dog_list = ['dog.{0}.jpg'.format(i) for i in range(train_num+val_num, train_num+val_num+test_num)]

    # execute separate
    # separate train dataset
    separateDataset(train_cat_dir, train_dog_dir, train_cat_list, train_dog_list)
    # separate validation dataset
    separateDataset(val_cat_dir, val_dog_dir, val_cat_list, val_dog_list)
    # separate validation dataset
    separateDataset(test_cat_dir, test_dog_dir, test_cat_list, test_dog_list)


def separateDataset(cat_dst_dir, dog_dst_dir, cat_frame_list, dog_frame_list):
    """
    sepatate dataset
    :param cat_dst_dir:
    :param dog_dst_dir:
    :param cat_frame_list:
    :param dog_frame_list:
    :return:
    """
    for cat_frame, dog_frame in zip(cat_frame_list, dog_frame_list):
        cat_src = os.path.join(original_dataset_dir, cat_frame)
        dog_src = os.path.join(original_dataset_dir, dog_frame)
        cat_dst = os.path.join(cat_dst_dir, cat_frame)
        dog_dst = os.path.join(dog_dst_dir, dog_frame)
        shutil.copy(cat_src, cat_dst)
        shutil.copy(dog_src, dog_dst)

# def vgg16Net():
#     model = models.Sequential()
#     model.add()


if __name__ == "__main__":
    # 构建数据
    # structureDataset()
    # 获取数据信息
    train_cat_list = os.listdir(train_cat_dir)
    train_dog_list = os.listdir(train_dog_dir)
    val_cat_list = os.listdir(val_cat_dir)
    val_dog_list = os.listdir(val_dog_dir)
    test_cat_list = os.listdir(test_cat_dir)
    test_dog_list = os.listdir(test_dog_dir)
    print(len(train_cat_list), len(train_dog_list))
    print(len(val_cat_list), len(val_dog_list))
    print(len(test_cat_list), len(test_dog_list))

