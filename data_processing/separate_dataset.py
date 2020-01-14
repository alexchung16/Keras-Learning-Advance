#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File separate_dataset.py
# @ Description
# @ Author alexchung
# @ Time 14/10/2019 PM 15:15

import os
import pickle
import numpy as np
import shutil


# origin dataset
original_dataset_dir = '/home/alex/Documents/datasets/flower_photos'
# separate dataset
base_dir = '/home/alex/Documents/datasets/flower_photos_separate'
# train dataset
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

def makedir(path):
    """
    create dir
    :param path:
    :return:
    """
    if os.path.exists(path):
        print('{0} is exist'.format(path))
    else:
        try:
            os.makedirs(path)
            print('{0} has been created'.format(path))
        except FileNotFoundError as e:
            print(e)


def separateDataset(origin_path='', separate_path='', separate_ratio=0.8):
    """
    separate
    :param origin_path:
    :param separate_path:;
    :param separate_ratio:
    :return:
    """
    train_dir = os.path.join(separate_path, 'train')
    test_dir = os.path.join(separate_path, 'test')
    makedir(train_dir)
    makedir(test_dir)
    classes_list = os.listdir(origin_path)
    for index, class_name in enumerate(classes_list):
        class_path = os.path.join(origin_path, class_name)
        # train dataset path
        train_dataset_path = os.path.join(train_dir, class_name)
        # test dataset path
        test_dataset_path = os.path.join(test_dir, class_name)
        makedir(train_dataset_path)
        makedir(test_dataset_path)
        class_img_list = os.listdir(class_path)
        train_num = int(len(class_img_list) * separate_ratio)
        test_num = len(class_img_list) - train_num
        for index, file_name in enumerate(class_img_list):
            if index < train_num:
                train_src = os.path.join(class_path, file_name)
                train_dst = os.path.join(train_dataset_path, file_name)
                shutil.copy(train_src, train_dst)
            else:
                test_src = os.path.join(class_path, file_name)
                test_dst = os.path.join(test_dataset_path, file_name)
                shutil.copy(test_src, test_dst)



if __name__ == "__main__":

    # # construct label dataset
    # constructLabelDataset()

    # separate dataset
    separateDataset(origin_path=original_dataset_dir, separate_path=base_dir, separate_ratio=0.8)


