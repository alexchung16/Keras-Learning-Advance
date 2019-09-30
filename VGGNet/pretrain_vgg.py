#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File pretrian_vgg.py
# @ Description
# @ Author alexchung
# @ Time 30/9/2019 AM 10:54


import os
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras import models, layers
from keras import optimizers, losses
from keras_preprocessing.image import ImageDataGenerator


# data path
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

# vgg16 base model
# include_top=False : do not contain top layer(Dense layer)
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# 生成器做像素尺度变化
data_generate = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_feature(datasets_dir, sample_count):
    """
    according conv_base to extract base
    :param datasets:
    :param sample_count:
    :return:
    """
    feature = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count, ))

    generate = data_generate.flow_from_directory(
        directory=datasets_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
    )
    # batch num
    i = 0
    for image_batch, labels_batch in generate:
        # predict to (4, 4, 152)
        feature_batch = conv_base.predict(image_batch)
        feature[i*batch_size: (1+i)*batch_size] = feature_batch
        labels[i*batch_size: (1+i)*batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return feature, labels

def subDenseNet():
    """
    construct sub Net only constructed by dense layer
    :return:
    """
    model = models.Sequential()
    model.add(layers.Dense(units=256, activation='relu', input_shape=(4*4*512, )))
    model.add(layers.Dropout(rate=0.5))
    model.add(layers.Dense(units=1, activation='sigmoid'))
    return model

def plotTrainValidationLossAccuracy(history):
    """
    Training validation loss and accuracy of epoch
    :param history: training parameter
    :return:
    """
    # history.keys() = ['val_loss', 'val_binary_accuracy', 'loss', 'binary_accuracy']
    history_dict = history.history
    loss = history_dict['loss']
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(loss) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.plot(epochs, loss, 'bo', label='Training loss')
    ax1.plot(epochs, val_loss, 'b', label='Validation loss')
    ax1.set_title('Training and validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, acc, 'bo', label='Training accuracy')
    ax2.plot(epochs, val_acc, 'b', label='Validation accuracy')
    ax2.set_title('Training and validation accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.show()

if __name__ == "__main__":

    train_feature, train_labels = extract_feature(train_dir, 2000)
    val_feature, val_labels = extract_feature(val_dir, 1000)
    test_feature, test_labels = extract_feature(test_dir, 1000)

    print(train_feature.shape)

    # flat dataset as new datasets
    train_feature = np.reshape(train_feature, (2000, 4*4*512))
    val_feature = np.reshape(val_feature, (1000, 4*4*512))
    test_feature = np.reshape(test_feature, (1000, 4 * 4 * 512))

    # struct dense layer
    model = subDenseNet()

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss = losses.binary_crossentropy,
                  metrics=['accuracy'])
    history = model.fit(x=train_feature, y=train_labels, batch_size=20, epochs=30,
              validation_data=(val_feature, val_labels))

    plotTrainValidationLossAccuracy(history)


