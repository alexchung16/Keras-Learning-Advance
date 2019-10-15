#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File separate_binary_databases.py
# @ Description
# @ Author alexchung
# @ Time 15/10/2019 PM 15:20


import os
import pickle
import cv2 as cv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import struct

# origin dataset
original_dataset_dir = '/home/alex/Documents/datasets/dogs-vs-cats/train'
# separate dataset
base_dir = '/home/alex/Documents/datasets/dogs_cat_binary_separate'
# train dataset
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

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
        makedir(test_dir)
except FileNotFoundError as e:
    print(e)


def binaryRecord(image_data, label_data):
    """
    construct binary record
    :param image_data:
    :param label_data:
    :return:
    """
    # convert array to bytes
    byte_img = image_data.tobytes()
    # pac index to signedchar(integer)
    byte_index = struct.pack('b', label_data)
    # contact image bytes and index bytes
    record_byte = byte_index + byte_img
    return record_byte


def constructBinaryDataset(origin_dir, bin_dir, name_list, label_list, class_name=None,
                           img_length=224, img_width=224):
    """
    construct binary dataset
    :param origin_path:
    :param train_bin_path:
    :param test_bin_path:
    :param train_num:
    :param test_num:
    :param class_name:
    :return:
    """
    if class_name is None:
        class_name = ['cat', 'dog']

    # binary file save path
    bin_path = os.path.join(bin_dir, 'image.bin')
    # meta info save path
    meta_path = os.path.join(bin_dir, 'meta.txt')

    with open(bin_path, 'ab+') as fa:
        for img_name, label_name in zip(name_list, label_list):
            # image path
            img_path = os.path.join(origin_dir, img_name)
            # get image
            raw_img = cv.imread(img_path)
            # reshape image shape to []224, 224, 3]
            reshape_img = cv.resize(src=raw_img, dsize=(img_length, img_width))
            # transpose image shape
            trans_img = np.transpose(reshape_img, (2, 0, 1))
            # get label index
            index = class_name.index(label_name)
            # get record bytes
            record_byte = binaryRecord(trans_img, index)
            fa.write(record_byte)
        fa.close()

    with open(meta_path, 'w') as fw:
        for i, name in enumerate(class_name):
            fw.write(name + '\n')
        fw.close()

if __name__ == "__main__":

    LABEL_LENGTH = 1
    IMAGE_LENGTH = 224
    IMAGE_WIDTH = 224
    IMAGE_CHANNEL = 3

    image_length = IMAGE_LENGTH * IMAGE_WIDTH * IMAGE_CHANNEL

    class_name = ['cat', 'dog']
    train_num = 1000
    test_num = 500

    train_name_list = []
    train_label_list = []
    test_name_list = []
    test_label_list = []
    for n in class_name:
        train_name_list += ['{0}.{1}.jpg'.format(n, i) for i in range(0, train_num)]
        train_label_list += [n for i in range(0, train_num)]
        test_name_list += ['{0}.{1}.jpg'.format(n, i) for i in range(train_num, train_num + test_num)]
        test_label_list += [n for i in range(train_num, train_num + test_num)]

    constructBinaryDataset(original_dataset_dir, train_dir, train_name_list, train_label_list,
                           class_name, IMAGE_LENGTH, IMAGE_WIDTH)
    constructBinaryDataset(original_dataset_dir, test_dir, test_name_list, test_label_list,
                           class_name, IMAGE_LENGTH, IMAGE_WIDTH)

    # constructBinaryDataset(original_dataset_dir, train_dir,test_dir)










