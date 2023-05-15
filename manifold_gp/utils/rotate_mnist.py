#!/usr/bin/env python
# encoding: utf-8

import requests
import gzip
import numpy as np
from scipy import ndimage


class RotatedMNIST():
    def __init__(self) -> None:
        self.train_x = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        self.train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
        self.test_x = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
        self.test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

    def generate_trainset(self, rows, cols, dataset=None, save=False):
        if dataset is None:
            # Inputs
            request = requests.get(self.train_x)
            open("/tmp/train_x.gz", "wb").write(request.content)
            with gzip.open("/tmp/train_x.gz") as bytestream:
                bytestream.read(16)
                buf = bytestream.read(28 * 28 * 60000 * 1)
                X = self.scale(np.frombuffer(buf, dtype=np.uint8).astype(
                    np.float32)).reshape(60000, 28, 28)

            # Labels
            request = requests.get(self.train_labels)
            open("/tmp/train_labels.gz", "wb").write(request.content)
            with gzip.open("/tmp/train_labels.gz") as bytestream:
                bytestream.read(8)
                buf = bytestream.read(1 * 60000)
                labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        else:
            X, labels = dataset

        # Outputs
        y = np.random.uniform(low=-45, high=45, size=(rows, cols))

        # Dataset
        train_x = np.zeros((rows*(cols+1), 28, 28))
        train_y = np.zeros((rows*(cols+1),))
        train_labels = np.zeros((rows*(cols+1),))

        for i in range(y.shape[0]):
            train_x[i*(cols+1), :, :] = X[i, :, :]
            train_labels[i*(cols+1)] = labels[i]
            train_y[i*(cols+1)] = 0.0

            for j in range(y.shape[1]):
                train_x[i*(cols+1) + j + 1, :, :] = ndimage.rotate(
                    X[i, :, :], y[i, j], reshape=False, cval=-0.5)
                train_labels[i*(cols+1) + j + 1] = labels[i]
                train_y[i*(cols+1) + j + 1] = y[i, j]
        train_x = train_x.reshape([rows*(cols+1), -1])

        if save:
            np.savetxt('benchmarks/datasets/mnist_x_train.csv', train_x)
            np.savetxt('benchmarks/datasets/mnist_y_train.csv', train_y)
            np.savetxt('benchmarks/datasets/mnist_label_train.csv', train_labels)

        return train_x, train_y, train_labels

    def generate_testset(self, rows, cols, dataset=None, save=False):
        if dataset is None:
            # Inputs
            request = requests.get(self.test_x)
            open("/tmp/test_x.gz", "wb").write(request.content)
            with gzip.open("/tmp/test_x.gz") as bytestream:
                bytestream.read(16)
                buf = bytestream.read(28 * 28 * 10000 * 1)
                X = self.scale(np.frombuffer(buf, dtype=np.uint8).astype(
                    np.float32)).reshape(10000, 28, 28)

            # Labels
            request = requests.get(self.test_labels)
            open("/tmp/test_labels.gz", "wb").write(request.content)
            with gzip.open("/tmp/test_labels.gz") as bytestream:
                bytestream.read(8)
                buf = bytestream.read(1 * 10000)
                labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        else:
            X, labels = dataset

        # Outputs
        y = np.random.uniform(low=-45, high=45, size=(rows, cols))

        # Dataset
        test_x = np.zeros((rows*(cols+1), 28, 28))
        test_y = np.zeros((rows*(cols+1),))
        test_labels = np.zeros((rows*(cols+1),))

        for i in range(y.shape[0]):
            test_x[i*(cols+1), :, :] = X[i, :, :]
            test_labels[i*(cols+1)] = labels[i]
            test_y[i*(cols+1)] = 0.0

            for j in range(y.shape[1]):
                test_x[i*(cols+1) + j + 1, :, :] = ndimage.rotate(
                    X[i, :, :], y[i, j], reshape=False, cval=-0.5)
                test_labels[i*(cols+1) + j + 1] = labels[i]
                test_y[i*(cols+1) + j + 1] = y[i, j]
        test_x = test_x.reshape([rows*(cols+1), -1])

        if save:
            np.savetxt('benchmarks/datasets/mnist_x_test.csv', test_x)
            np.savetxt('benchmarks/datasets/mnist_y_test.csv', test_y)
            np.savetxt('benchmarks/datasets/mnist_label_test.csv', test_labels)

        return test_x, test_y, test_labels

    def scale(self, X):
        return (X - (255 / 2.0)) / 255

    def rescale(self, X):
        return (X*255) + (255 / 2.0)

    def rad_to_deg(self, X):
        return X * 180 / np.pi

    def deg_to_rad(self, X):
        return X * np.pi / 180
