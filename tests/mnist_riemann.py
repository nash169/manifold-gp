# !/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import gpytorch
import random
import matplotlib.pyplot as plt

from importlib.resources import files

from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.models.riemann_gp import RiemannGP

# Set device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Train Dataset
train_x = torch.from_numpy(np.loadtxt(
    files('manifold_gp.data').joinpath('mnist_x_train.csv'))).float().to(device)
train_y = torch.from_numpy(np.loadtxt(
    files('manifold_gp.data').joinpath('mnist_y_train.csv'))).float().to(device)
train_label = torch.from_numpy(np.loadtxt(
    files('manifold_gp.data').joinpath('mnist_label_train.csv'))).float().to(device)

# Test Dataset
test_x = torch.from_numpy(np.loadtxt(
    files('manifold_gp.data').joinpath('mnist_x_test.csv'))).float().to(device)
test_y = torch.from_numpy(np.loadtxt(
    files('manifold_gp.data').joinpath('mnist_y_test.csv'))).float().to(device)
test_label = torch.from_numpy(np.loadtxt(
    files('manifold_gp.data').joinpath('mnist_label_test.csv'))).float()

# Generate Dataset
# mnist = RotatedMNIST()
# train_x += 0.5 # mnist.rescale(train_x)
# test_x += 0.5 # mnist.rescale(test_x)
# train_y = mnist.deg_to_rad(train_y)
# test_y = mnist.deg_to_rad(test_y)

# Remove digit
digits = [0]
for digit in digits:
    train_x = train_x[train_label != digit]
    train_y = train_y[train_label != digit]
    train_label = train_label[train_label != digit]
    test_x = test_x[test_label != digit]
    test_y = test_y[test_label != digit]
    test_label = test_label[test_label != digit]

# Initialize kernel
nu = 3
neighbors = 50
modes = 300
support_kernel = gpytorch.kernels.RBFKernel()
kernel = gpytorch.kernels.ScaleKernel(RiemannMaternKernel(nu=nu, nodes=train_x, neighbors=neighbors, modes=modes, support_kernel=support_kernel))

# Training parameters
lr = 1e-2
iters = 1000
verbose = True

# Loop
count = 1
samples = 1
stats = torch.zeros(10, 2)
loss = torch.zeros(samples, 1)

for i in range(samples):
    print(f"Iteration: {count}/{samples}")

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-8))
    model = RiemannGP(train_x, train_y, likelihood, kernel).to(device)

    # Model Hyperparameters
    hypers = {
        'likelihood.noise_covar.noise': 1e-4,
        'covar_module.base_kernel.epsilon': random.uniform(1, 3),
        'covar_module.base_kernel.lengthscale': random.uniform(1, 3),
        'covar_module.outputscale': random.uniform(0.5, 1),
        'covar_module.base_kernel.support_kernel.lengthscale': 0.5
    }
    model.initialize(**hypers)

    # Train model
    loss[i] = model.manifold_informed_train(lr, iters, verbose)
    # loss = model.vanilla_train(lr[1], iters, verbose)

    # Model Evaluation
    likelihood.eval()
    model.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(test_x))
        error = (test_y - preds.mean).cpu()
        std = preds.stddev.cpu()

        for j in range(10):
            if torch.any(test_label == j) == True:
                stats[j, 0] += error[test_label == j].abs().sum() / error[test_label == j].shape[0]
                stats[j, 1] += std[test_label == j].abs().sum() / std[test_label == j].shape[0]
            else:
                stats[j, 0] += 0
                stats[j, 1] += 0

        # Largest error
        idx_err = torch.argsort(error.abs(), descending=True)
        fig = plt.figure()
        fig.subplots_adjust(wspace=1.2)
        for i in range(6):
            ax = fig.add_subplot(2, 3, i+1)
            ax.imshow(test_x[idx_err[i], :].cpu().numpy().reshape(28, 28, order='F'), cmap='gray')

            ax.set_title('Label: ' + str(test_label.cpu().numpy()[idx_err[i]]) + ' Index: ' + str(idx_err.cpu().numpy()[i]) +
                         '\nTruth: ' + str(test_y.cpu().numpy()[idx_err[i]]) + '\nGP: ' + str(preds.mean.cpu().numpy()[idx_err[i]]) +
                         '\nError: ' + str(error[idx_err[i]].cpu().numpy()) + '\nStd: ' + str(preds.stddev.cpu().numpy()[idx_err[i]]), fontsize=10)
        fig.savefig('outputs/riemann_'+str(nu)+'_'+str(neighbors) + '_err_' + str(count) + '.png')

        # Largest standard deviation
        idx_std = torch.argsort(std, descending=True)
        fig = plt.figure()
        fig.subplots_adjust(wspace=1.0)
        for i in range(6):
            ax = fig.add_subplot(2, 3, i+1)
            ax.imshow(test_x[idx_std[i], :].cpu().numpy().reshape(28, 28, order='F'), cmap='gray')
            ax.set_title('Label: ' + str(test_label.cpu().numpy()[idx_std[i]]) + ' Index: ' + str(idx_std.cpu().numpy()[i]) +
                         '\nTruth: ' + str(test_y.cpu().numpy()[idx_std[i]]) + '\nGP: ' + str(preds.mean.cpu().numpy()[idx_std[i]]) +
                         '\nError: ' + str(error[idx_std[i]].cpu().numpy()) + '\nStd: ' + str(preds.stddev.cpu().numpy()[idx_std[i]]), fontsize=10)
        fig.savefig('outputs/riemann_'+str(nu)+'_'+str(neighbors) + '_std_' + str(count) + '.png')

    count += 1

stats /= samples
# stats = stats * 180 / torch.pi
results = torch.cat((stats[:, 0].unsqueeze(-1), (stats[:, 0] + stats[:, 1]).unsqueeze(-1),
                    (stats[:, 0] - stats[:, 1]).unsqueeze(-1), stats[:, 1].unsqueeze(-1), loss), dim=1)
np.savetxt('outputs/riemann_'+str(nu)+'_'+str(neighbors) + '.csv', results.detach().numpy())

# mse = torch.linalg.norm(test_y - preds.mean)/test_y.shape[0]

# import matplotlib.pyplot as plt
# from keras.datasets import mnist

# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# for i in range(20):
#     fig = plt.figure(figsize=(5, 5))
#     ax = fig.add_subplot(111)
#     ax.imshow(X_train[i], cmap='gray')
#     ax.tick_params(axis='both', which='both', bottom=False,
#                    labelbottom=False, left=False, labelleft=False)
#     # ax.set_title('Label: ' + str(y_train[0]), fontsize=30)
#     fig.savefig('mnist_' + str(y_train[i]) + '.png')