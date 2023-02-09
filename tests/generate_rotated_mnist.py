import requests
import gzip
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from manifold_gp.utils.rotate_mnist import RotatedMNIST
from numpy.random import default_rng

obj = RotatedMNIST()
train_x, train_y, train_labels = obj.generate_trainset(100, 50, True)
test_x, test_y, test_labels = obj.generate_testset(100, 20, True)

rng = default_rng()
sample = rng.choice(train_x.shape[0], size=12, replace=False)
# sample = range(12)
count = 1

fig = plt.figure()
fig.subplots_adjust(wspace=1.2)
for i in sample:
    ax = fig.add_subplot(3, 4, count)
    ax.imshow(obj.rescale(train_x[i, :].reshape(28, 28)), cmap='gray')
    ax.set_title(
        'Label: ' + str(train_labels[i]) + ' Truth: ' + str(train_y[i]), fontsize=10)
    count += 1
plt.show()

# response = requests.get(
#     'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
# open("temp.gz", "wb").write(response.content)


# # unzip data
# with gzip.open("temp.gz") as bytestream:
#     bytestream.read(16)
#     buf = bytestream.read(28 * 28 * 10000 * 1)
#     data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
#     data = (data - (255 / 2.0)) / 255  # rescaling value to [-0.5,0.5]
#     data = data.reshape(10000, 28, 28, 1)  # reshape into tensor
#     data = np.reshape(data, [10000, -1])

# image = np.reshape(data[0, :], (-1, 28))

# angle = 90  # np.random.randint(-90, 90, 1)[0]
# bg_value = -0.5
# new_img = ndimage.rotate(image, angle, reshape=False, cval=bg_value)

# image1 = (image*255) + (255 / 2.0)
# new_img1 = (new_img*255) + (255 / 2.0)

# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(image1, cmap='gray')
# ax = fig.add_subplot(1, 2, 2)
# ax.imshow(new_img1, cmap='gray')
# plt.show()
