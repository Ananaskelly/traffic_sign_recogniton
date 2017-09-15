import numpy as np
import collections
import os
import img_processing
import csv
import configparser as cfp

CSV_NAME_TRAIN = 'gt_train.csv'
CSV_NAME_TEST = 'gt_test.csv'

np.set_printoptions(threshold=np.nan)


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


def read(num_classes):

    config = cfp.RawConfigParser()
    config.read('config.cfg')

    path_to_data_train = config.get('path', 'data_russian_dir_train')
    path_to_data_test = config.get('path', 'data_russian_dir_test')
    main_path = config.get('path', 'data_russian_main_dir')

    num_training = sum([len(list(filter(lambda f: f.endswith('.png'), files))) for r, d, files in
                       os.walk(path_to_data_train)])
    num_test = sum([len(list(filter(lambda f: f.endswith('.png'), files))) for r, d, files in
                       os.walk(path_to_data_train)])

    train_images = np.zeros((num_training, 32, 32, 3))
    test_images = np.zeros((num_test, 32, 32, 3))
    labels_train = np.zeros((num_training), dtype='int32')
    labels_test = np.zeros((num_test), dtype='int32')

    counter = 0
    with open(os.path.join(main_path, CSV_NAME_TRAIN), 'r', newline='') as csv_file:
        data_reader = csv.reader(csv_file, delimiter=',')
        it = iter(data_reader)
        next(it)
        for row in it:
            filename = row[0]
            class_no = row[1]
            train_images[counter] = img_processing.load_and_process_without_roi(os.path.join(path_to_data_train,
                                                                                           filename), False)
            labels_train[counter] = class_no
            counter += 1
            if counter%1000 == 0:
                print(str(counter)+'images processed')
    counter = 0
    with open(os.path.join(main_path, CSV_NAME_TEST), 'r', newline='') as csv_file:
        data_reader = csv.reader(csv_file, delimiter=',')
        it = iter(data_reader)
        next(it)
        for row in it:
            filename = row[0]
            class_no = row[1]
            test_images[counter] = img_processing.load_and_process_without_roi(os.path.join(path_to_data_test,
                                                                                           filename), False)
            labels_test[counter] = class_no
            counter += 1
            if counter%1000 == 0:
                print(str(counter)+'images processed')
    perm = np.arange(num_training)
    np.random.shuffle(perm)
    train_images = train_images[perm]
    labels_train = labels_train[perm]

    perm = np.arange(num_test)
    np.random.shuffle(perm)
    test_images = test_images[perm]
    labels_test = labels_test[perm]

    train_labels = dense_to_one_hot(labels_train, num_classes)
    test_labels = dense_to_one_hot(labels_test, num_classes)

    train = DataSet(train_images, train_labels)

    test = DataSet(test_images[:10000], test_labels[:10000])
    ds = collections.namedtuple('Datasets', ['train', 'test'])

    return ds(train=train, test=test)


def dense_to_one_hot(labels_dense, num_classes):
    labels_one_hot = np.zeros(shape=(len(labels_dense), num_classes),)
    for i in range(len(labels_dense)):
        labels_one_hot.itemset((i, labels_dense[i]), 1)
    return labels_one_hot