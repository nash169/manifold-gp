#!/usr/bin/env python
# encoding: utf-8

import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy import ndimage


def rotate_mnist(samples, labels, num_samples, rots_sample, shuffle=False):
    rotations = np.random.uniform(low=-45, high=45, size=(num_samples, rots_sample))
    sampled_x = np.zeros((num_samples*(rots_sample+1), 28, 28))
    sampled_y = np.zeros((num_samples*(rots_sample+1),))
    sampled_labels = np.zeros((num_samples*(rots_sample+1),))

    for i in range(num_samples):
        sampled_x[i*(rots_sample+1), :, :] = samples[i, :, :]
        sampled_y[i*(rots_sample+1)] = 0.0
        sampled_labels[i*(rots_sample+1)] = labels[i]

        for j in range(rots_sample):
            sampled_x[i*(rots_sample+1) + j + 1, :, :] = ndimage.rotate(samples[i, :, :], rotations[i, j], reshape=False)
            sampled_y[i*(rots_sample+1) + j + 1] = rotations[i, j]
            sampled_labels[i*(rots_sample+1) + j + 1] = labels[i]
    
    if shuffle:
        rand_idx = np.random.permutation(num_samples*(rots_sample+1))
        sampled_x, sampled_y, sampled_labels = sampled_x[rand_idx], sampled_y[rand_idx], sampled_labels[rand_idx]

    return sampled_x, sampled_y, sampled_labels


if __name__ == "__main__":
    single_digit = True if len(sys.argv) > 1 and sys.argv[1] == "single" else False
    (train_samples, train_labels), (test_samples, test_labels) = tf.keras.datasets.mnist.load_data()

    if single_digit:
        digits_idx = [1, 8, 5, 7, 2, 0, 18, 15, 17, 4]  # 9, 7, 4, 5, 6, 0, 3, 1, 2, 8
        sampled_x, sampled_y, sampled_labels = rotate_mnist(train_samples[digits_idx], train_labels[digits_idx], num_samples=len(digits_idx), rots_sample=0)
    else:
        sampled_x, sampled_y, sampled_labels = rotate_mnist(train_samples, train_labels, num_samples=100, rots_sample=100)
        # test_x, test_y, test_labels = rotate_mnist(test_samples, test_labels, num_samples=100, rots_sample=10)

    # rand_idx = np.random.permutation(sampled_x.shape[0])
    # sampled_x, sampled_y, sampled_labels = sampled_x[rand_idx], sampled_y[rand_idx], sampled_labels[rand_idx]

    fig, axs = plt.subplots(2, 5)
    for i, index in enumerate(np.random.permutation(np.arange(sampled_x.shape[0]))[:10]):
        axs[0 if i <= 4 else 1, i if i <= 4 else i-5].imshow(sampled_x[index], cmap='gray')
        axs[0 if i <= 4 else 1, i if i <= 4 else i-5].tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)
        axs[0 if i <= 4 else 1, i if i <= 4 else i-5].set_title('Label: ' + str(sampled_labels[index]) + ' Rotation: ' + str(sampled_y[index]))
    fig.tight_layout()
    plt.show(block=False)
