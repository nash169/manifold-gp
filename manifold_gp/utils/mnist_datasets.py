#!/usr/bin/env python
# encoding: utf-8

import requests
import gzip
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


class RotatedMNIST():
    def __init__(self) -> None:
        self.train_x = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
        self.train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
        self.test_x = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
        self.test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

    def generate_trainset(self, rows, cols, dataset=None):
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

        return train_x, train_y, train_labels

    def generate_testset(self, rows, cols, dataset=None):
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

        return test_x, test_y, test_labels

    def scale(self, X):
        return (X - (255 / 2.0)) / 255

    def rescale(self, X):
        return (X*255) + (255 / 2.0)

    def rad_to_deg(self, X):
        return X * 180 / np.pi

    def deg_to_rad(self, X):
        return X * np.pi / 180


if __name__ == "__main__":
    visualize = False
    single_digit = False
    rot_mnist = RotatedMNIST()

    if single_digit:
        digits, _, digits_label = rot_mnist.generate_trainset(20, 0)
        idx = [1, 8, 5, 7, 2, 0, 18, 15, 17, 4]
        digits = digits[idx]
        digits_labels = digits_label[idx]

        x, y, labels = rot_mnist.generate_trainset(digits.shape[0], 1100, dataset=(digits.reshape(-1, 28, 28), digits_labels))
        # test_x, test_y, test_labels = rot_mnist.generate_trainset(digits.shape[0], 100, dataset=(digits.reshape(-1, 28, 28), digits_labels))
        idx = np.random.permutation(x.shape[0])
        x, y, labels = x[idx], y[idx], labels[idx]

        with open('datasets/mnist_single.npy', 'wb') as f:
            np.save(f, np.append(x, y[:, np.newaxis], axis=1))

        with open('datasets/mnist_single_labels.npy', 'wb') as f:
            np.save(f, labels[:, np.newaxis])

        # with open('outputs/datasets/mnist_single_test.npy', 'wb') as f:
        #     np.save(f, np.concatenate((test_labels[:, np.newaxis], test_y[:, np.newaxis], test_x), axis=1))
    else:
        x, y, labels = rot_mnist.generate_trainset(1000, 110)
        # test_x, test_y, test_labels = rot_mnist.generate_testset(100, 100)
        idx = np.random.permutation(x.shape[0])
        x, y, labels = x[idx], y[idx], labels[idx]

        with open('datasets/mnist.npy', 'wb') as f:
            np.save(f, np.append(x, y[:, np.newaxis], axis=1))

        with open('datasets/mnist_labels.npy', 'wb') as f:
            np.save(f, labels[:, np.newaxis])

        # with open('outputs/datasets/mnist_test.npy', 'wb') as f:
        #     np.save(f, np.concatenate((test_labels[:, np.newaxis], test_y[:, np.newaxis], test_x), axis=1))

    if visualize:
        # train
        indices = np.random.permutation(np.arange(x.shape[0]))[:10]
        fig, axs = plt.subplots(2, 5)
        for i, index in enumerate(indices):
            axs[0 if i <= 4 else 1, i if i <= 4 else i-5].imshow(x[index, :].reshape(28, 28), cmap='gray')
            axs[0 if i <= 4 else 1, i if i <= 4 else i-5].tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)
            axs[0 if i <= 4 else 1, i if i <= 4 else i-5].set_title('Label: ' + str(labels[index]) + ' Rotation: ' + str(y[index]))  # + ' ID: ' + str(index)
        # fig.suptitle('Train')
        fig.tight_layout()

        # # test
        # indices = np.random.permutation(np.arange(test_x.shape[0]))[:10]
        # fig, axs = plt.subplots(2, 5)
        # for i, index in enumerate(indices):
        #     axs[0 if i <= 4 else 1, i if i <= 4 else i-5].imshow(test_x[index, :].reshape(28, 28), cmap='gray')
        #     axs[0 if i <= 4 else 1, i if i <= 4 else i-5].tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)
        #     axs[0 if i <= 4 else 1, i if i <= 4 else i-5].set_title('Label: ' + str(test_labels[index]) + ' Rotation: ' + str(test_y[index]))  # + ' ID: ' + str(index)
        # fig.suptitle('Test')
        # fig.tight_layout()

    plt.show()
