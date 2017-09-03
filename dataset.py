import numpy as np
import collections
import os
import proc
import glob
import cv2
import csv
import configparser as cfp
from tensorflow.python.framework import dtypes

CSV_NAME_PREF = 'GT-'
CSV_NAME_EX = '.csv'

class DataSet(object):

    def __init__(self,
                 images,
                 labels):

        self._images = images
        self._num_examples = images.shape[0]
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            # shuffle
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end]


def read_gtrsb_dataset():

    config = cfp.RawConfigParser()
    config.read('config.cfg')

    path_to_data = config.get('path', 'data_dir')

    num_all = sum([len(list(filter(lambda f: f.endswith('.ppm'), files))) for r, d, files in os.walk(path_to_data)])
    num_training = round(num_all*0.9)
    num_test = num_all - num_training

    all_images = np.zeros((num_all, 32, 32))

    dirs = os.listdir(path_to_data)
    counter = 0

    labels_array = np.asarray([])
    num_classes = len(dirs)
    for ind, d in enumerate(dirs):
        files = os.listdir(os.path.join(path_to_data, d))
        labels_curr = np.full(len(list(filter(lambda f: f.endswith('.ppm'), files))), ind, 'int32')
        labels_array = np.concatenate((labels_array, labels_curr))
        with open(os.path.join(path_to_data, d, CSV_NAME_PREF + d + CSV_NAME_EX), 'r', newline='') as csv_file:
            data_reader = csv.reader(csv_file, delimiter=';')
            it = iter(data_reader)
            next(it)
            for row in it:
                all_images[counter] = proc.load_and_process(os.path.join(path_to_data, d, row[0]), int(row[1]),
                                                            int(row[2]), int(row[3]), int(row[4]))
        print(str(ind) + ' folder loaded')

    all_labels = dense_to_one_hot(labels_array, num_classes)

    perm = np.arange(num_all)
    np.random.shuffle(perm)
    all_images = all_images[perm]
    all_labels = all_labels[perm]
    mask = range(num_training)
    train_images = all_images[mask]
    train_labels = all_labels[mask]

    mask = range(num_training, num_training + num_test)
    test_images = all_images[mask]
    test_labels = all_labels[mask]

    train = DataSet(train_images, train_labels)

    test = DataSet(test_images, test_labels)
    ds = collections.namedtuple('Datasets', ['train', 'test'])

    return ds(train=train, test=test)


def dense_to_one_hot(labels_dense, num_classes):
    labels_one_hot = np.zeros(shape=(len(labels_dense), num_classes),)
    for i in range(len(labels_dense)):
        labels_one_hot.itemset((i, labels_dense[i]), 1)
    return labels_one_hot
