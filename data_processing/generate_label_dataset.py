#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : generate_label_dataset.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/1/14 PM 2:17
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import shutil


# origin dataset
original_dataset_dir = '/home/alex/Documents/dataset/flower_perate'
# separate dataset
label_file_dir = '/home/alex/Documents/dataset/flower_label_file'

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

def generateLabelDataset(origin_dir, label_file_dir, classes_name=None):
    """
    construct label dataset
    :param origin_path:
    :param separate_path:
    :param train_num:
    :param class_num:
    :return:
    """


    if classes_name is not None:
        classes_name = classes_name
    else:
        classes_name = [class_name for class_name in os.listdir(origin_dir)
                       if os.path.isdir(os.path.join(origin_dir, class_name))]

    # step 1 traversal all dataset
    img_name_list = []
    img_label_list = []
    for class_label, class_name in enumerate(sorted(classes_name)):
        img_class_path = os.path.join(origin_dir, class_name)
        for i, img_name in enumerate(os.listdir(img_class_path)):
            if os.path.isfile(os.path.join(img_class_path, img_name)):
                img_name_list.append(img_name)
                img_label_list.append(class_name)
    makedir(label_file_dir)
    # step write data to label file format
    label_file = os.path.join(label_file_dir, 'label.txt')

    with open(label_file, 'w') as fw:
        num_samples = 0
        for img_name, img_label in zip(img_name_list, img_label_list):
            try:
                train_src = os.path.join(os.path.join(origin_dir, img_label), img_name)
                train_dst = os.path.join(label_file_dir, img_name)
                shutil.copy(train_src, train_dst)
                fw.write(img_name + ',' + img_label + '\n')
                num_samples += 1
            except Exception as e:
                print(e)
                continue
        print('There are {0} samples been wrote to {1}'.format(num_samples, label_file_dir))
        fw.close()


if __name__ == "__main__":

    generateLabelDataset(origin_dir=original_dataset_dir,
                         label_file_dir=label_file_dir)


