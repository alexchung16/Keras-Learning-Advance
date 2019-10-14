#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File separate_data.py
# @ Description
# @ Author alexchung
# @ Time 14/10/2019 PM 15:15

import os
import pickle
import numpy as np
import shutil
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# origin dataset
original_dataset_dir = '/home/alex/Documents/datasets/dogs-vs-cats/train'
# separate dataset
base_dir = '/home/alex/Documents/datasets/dogs_cat_label_separate'
# train dataset
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# train_label.txt
train_label = os.path.join(train_dir, 'label.txt')
# test_label.txt
test_label = os.path.join(test_dir, 'label.txt')

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

def constructLabelDataset(origin_path='', separate_path='', train_num=1000, test_num=500, class_name=None):
    """
    construct label dataset
    :param origin_path:
    :param separate_path:
    :param train_num:
    :param class_num:
    :return:
    """

    if class_name is None:
        class_name = ['cat', 'dog']
    # construct list

    # train_list = [['{0}.{1}.jpg'.format(n, i) for i in range(0, train_num)] for n in class_name]
    # test_list = [['{0}.{1}.jpg'.format(n, i) for i in range(train_num, train_num + test_num)] for n in class_name]
    train_name_list = []
    train_label_list = []
    test_name_list = []
    test_label_list = []
    for n in class_name:
        train_name_list += ['{0}.{1}.jpg'.format(n, i) for i in range(0, train_num)]
        train_label_list += [n for i in range(0,train_num)]
        test_name_list += ['{0}.{1}.jpg'.format(n, i) for i in range(train_num, train_num+test_num)]
        test_label_list += [n for i in range(train_num, train_num+test_num)]


    with open(train_label, 'w') as fw:
        for file_name, label in zip(train_name_list, train_label_list):
            train_src = os.path.join(original_dataset_dir, file_name)
            train_dst = os.path.join(train_dir, file_name)
            shutil.copy(train_src, train_dst)
            fw.write(file_name + ',' + label + '\n')
        fw.close()

    with open(test_label, 'w') as fw:
        for file_name, label in zip(test_name_list, test_label_list):
            test_src = os.path.join(original_dataset_dir, file_name)
            test_dst = os.path.join(test_dir, file_name)
            shutil.copy(test_src, test_dst)
            fw.write(file_name + ',' + label + '\n')
        fw.close()



if __name__ == "__main__":

    # constructLabelDataset()
    # strip
    # split
    file_list = []
    class_list = []
    classes = []
    with open(train_label, 'r') as fr:
        all_lines = fr.readlines()
        # for line in all_line:
        #     class_list += line.strip('\n').split(',')
        for line_index, line_content in enumerate(all_lines):
            line_content = line_content.strip('\n').split(',')

            file_list.append(line_content[0])
            class_list.append(line_content[1])
            # class_list.append(classes.index(line_content[1]))
        fr.close()
    lf = LabelEncoder().fit(class_list)
    sparse_label = lf.transform(class_list).tolist()
    sparse_classes = lf.classes_
    print(sparse_label)
    print(sparse_classes)
    # label onehot encode
    class_array = np.array(class_list).reshape(len(class_list), 1)
    lof = OneHotEncoder().fit(class_array)
    onehot_label = lof.transform(class_array).toarray()
    print(onehot_label)
    # print(onehot_classes)




    # lf = LabelEncoder().fit(labels)
    # dataLabel = lf.transform(labels).tolist()


