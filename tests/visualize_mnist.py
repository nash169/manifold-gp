# !/usr/bin/env python
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from manifold_gp.utils.rotate_mnist import RotatedMNIST

# Generate Dataset
mnist = RotatedMNIST()

# num_samples = 20
# num_rotations = 0
train_x, train_y, train_labels = mnist.generate_trainset(20, 0)

# train_y = mnist.deg_to_rad()
# test_y = mnist.deg_to_rad()

# idx = [1, 2,  3, 4, 5,  6,  7,  8, 9]
# idx = [8, 5, 12, 2, 0, 18, 15, 17, 4]
idx = [8, 5, 12, 2, 0, 18, 15, 17, 4]

ref_x = train_x[idx, :].reshape(-1, 28, 28)
ref_labels = train_labels[idx]

# ref_x = train_x.reshape(-1, 28, 28)
# ref_labels = train_labels


train_x, train_y, train_labels = mnist.generate_trainset(len(ref_labels), 100, (ref_x, ref_labels), True)
test_x, test_y, test_labels = mnist.generate_testset(len(ref_labels), 50, (ref_x, ref_labels), True)


# data = test_x
# truth = test_y
# labels = test_labels
# for i in range(data.shape[0]):
#     fig = plt.figure(figsize=(5, 5))
#     ax = fig.add_subplot(111)
#     ax.imshow(data[i, :].reshape(28, 28), cmap='gray')
#     ax.tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)
#     ax.set_title('Label: ' + str(labels[i]) + ' Rotation: ' + str(truth[i]) + ' ID: ' + str(i))
#     fig.savefig('outputs/images/mnist_' + str(i) + '.png')
#     plt.close()

# plt.show()
