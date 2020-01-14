#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File generate_binary_databases.py
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
original_dataset_dir = '/home/alex/Documents/datasets/dogs_vs_cat_separate/train100'
# separate dataset
binary_dir = '/home/alex/Documents/datasets/dogs_cat_binary'
# train dataset
# train_dir = os.path.join(base_dir, 'train')
# test_dir = os.path.join(base_dir, 'test')

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
    byte_label = struct.pack('b', label_data)
    # contact image bytes and index bytes
    record_byte = byte_label + byte_img
    return record_byte


def constructBinaryDataset(origin_dir, bin_dir=None, classes_name=None, img_height=224, img_width=224):
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
    if classes_name is not None:
        classes_name = classes_name
    else:
        classes_name = [class_name for class_name in os.listdir(origin_dir)
                       if os.path.isdir(os.path.join(origin_dir, class_name))]

    classes_map = {}
    for i, class_name in enumerate(sorted(classes_name)):
        classes_map[class_name] = i

    if bin_dir is None:
        bin_dir = os.path.join(origin_dir, 'binary')
    makedir(bin_dir)
    # binary file save path
    bin_path = os.path.join(bin_dir, 'image.bin')
    # meta info save path
    meta_path = os.path.join(bin_dir, 'meta.txt')

    img_name_list = []
    img_label_list = []
    for class_name, class_label in classes_map.items():
        img_class_path = os.path.join(origin_dir, class_name)
        for i, img_name in enumerate(os.listdir(img_class_path)):
            img_path = os.path.join(img_class_path, img_name)
            if os.path.isfile(img_path):
                img_name_list.append(img_path)
                img_label_list.append(class_label)

    with open(bin_path, 'ab+') as fa:
        num_samples = 0
        for img_path, img_label in zip(img_name_list, img_label_list):
            try:
                # get image
                raw_img = cv.imread(img_path)
                # transfer image channel
                rgb_img = cv.cvtColor(raw_img, cv.COLOR_BGR2RGB)
                # reshape image shape to []224, 224, 3]
                reshape_img = cv.resize(src=rgb_img, dsize=(img_height, img_width))
                # transpose image shape
                trans_img = np.transpose(reshape_img, (2, 0, 1))
                # get label index
                # get record bytes
                record_byte = binaryRecord(trans_img, img_label)
                fa.write(record_byte)
                num_samples += 1
            except Exception as e:
                print('image {0} read failed due to {1}'.format(img_path, e))
        print('There are {0} samples been wrote to {1}'.format(num_samples, bin_path))
        fa.close()

    with open(meta_path, 'w') as fw:
        for i, name in enumerate(classes_map.keys()):
            fw.write(name + '\n')
        fw.close()


def unPicpkleBinary(bin_path, label_length, img_height, img_width, img_channel, label_forward=True):
    """
    unpickle binary file
    :param bin_path:
    :param label_length: label length bytes
    :param img_length: image length
    :param img_wigth: image width
    :param img_channel: image channel num
    :param label_head: label is head
    :return:
    """

    #--------------------------parse binary dataset-----------------------------------------
    # get bin file list
    file_list = os.listdir(bin_path)
    bin_list = [file for file in file_list if os.path.splitext(file)[1] == '.bin']
    image_vec_bytes = label_length + img_height * img_width * img_channel
    image_length = img_height * img_width * img_channel

    filename = []
    labels_array = np.zeros((0, label_length), dtype=np.uint8)
    images_array = np.zeros((0, image_length), dtype=np.uint8)
    for bin_file in bin_list:
        with open(os.path.join(bin_path, bin_file), 'rb') as f:
            bin_data = f.read()
        data = np.frombuffer(bin_data, dtype=np.uint8)
        data = data.reshape(-1, image_vec_bytes)
        # save label and image data
        if label_forward:
            label_image = np.hsplit(data, [label_length])
            label = label_image[0]
            image = label_image[1]
        else:
            image_label = np.hsplit(data, [image_length])
            label = image_label[1]
            image = image_label[0]
        # stack array
        labels_array = np.vstack((labels_array, label))
        images_array = np.vstack((images_array, image))
        filename.extend([os.path.join(bin_path, bin_file)] * data.shape[0])

    #---------------------------convert data from array to suitable format------------------------------
    images = []
    for i in range(images_array.shape[0]):
        reshape_img = np.reshape(images_array[i], (img_channel, img_height, img_width))
        trans_img = reshape_img.transpose((1, 2, 0))
        images.append(trans_img)

    # c
    labels = labels_array[:, 0].tolist()

    # read classes
    classes_map = {}
    with open(os.path.join(binary_dir, 'meta.txt'), 'r') as fr:
        all_lines = fr.readlines()
        for i, class_name in enumerate(all_lines):
            classes_map[class_name.strip('\n')] = len(classes_map)
        fr.close()

    return images, labels, filename, classes_map


if __name__ == "__main__":

    LABEL_LENGTH = 1
    IMAGE_HEIGHT = 224
    IMAGE_WIDTH = 224
    IMAGE_CHANNEL = 3

    image_length = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL


    constructBinaryDataset(origin_dir=original_dataset_dir,
                           bin_dir=binary_dir,
                           img_height=IMAGE_HEIGHT,
                           img_width=IMAGE_WIDTH)

    images, labels, filename, class_map = unPicpkleBinary(bin_path=binary_dir, label_length=LABEL_LENGTH,
                                                    img_height=IMAGE_HEIGHT, img_width=IMAGE_WIDTH,
                                                    img_channel=IMAGE_CHANNEL, label_forward=True)

    image = images[0]
    plt.imshow(image)
    plt.show()

    # constructBinaryDataset(original_dataset_dir, train_dir,test_dir)










