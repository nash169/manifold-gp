#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from importlib.resources import files
from mayavi import mlab
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.models.riemann_gp import RiemannGP
from manifold_gp.utils.generate_truth import groundtruth_from_mesh

# Set device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load mesh and generate ground truth
data_path = files('manifold_gp.data').joinpath('dragon10k.stl')
nodes, faces, truth = groundtruth_from_mesh(data_path)

# Generate ground truth
sampled_x = torch.from_numpy(nodes).float().to(device)
sampled_y = torch.from_numpy(truth).float().to(device)
(m, n) = sampled_x.shape

# Noisy dataset
manifold_noise = 0.0
noisy_x = sampled_x + manifold_noise * torch.randn(m, n).to(device)
function_noise = 0.01
noisy_y = sampled_y + function_noise * torch.randn(m).to(device)

# Train dataset
num_train = 500
perm_idx = torch.randperm(m)
train_idx = perm_idx[:num_train]
train_x = noisy_x[train_idx, :]
train_y = noisy_y[train_idx]

# Test points
num_test = 100
test_idx = torch.randperm(m)[:num_test]
test_x = noisy_x[test_idx, :]
test_y = noisy_y[test_idx]

# Initialize kernel
nu = 5
neighbors = 50
modes = 10
kernel = gpytorch.kernels.ScaleKernel(
    RiemannMaternKernel(sampled_x, nu, neighbors, modes))

# Initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = RiemannGP(sampled_x, sampled_y, likelihood, kernel).to(device)

# Train model
lr = 1e-1
iters = 10
verbose = True
model.manifold_informed_train(lr, iters, verbose)

# Model Evaluation
likelihood.eval()
model.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds = likelihood(model(sampled_x))

    # Mean
    mean = preds.mean

    # Standard deviation
    # std = preds.variance.sqrt()

    # One Posterior Sample
    posterior_sample = preds.sample()

    # Kernel evaluation
    kernel_eval = kernel(sampled_x[0, :].unsqueeze(
        0), sampled_x).evaluate().squeeze()

with torch.no_grad():
    # fig = plt.figure(figsize=(17, 9))

    # Bring data to cpu
    if use_cuda:
        sampled_x = sampled_x.cpu()
        sampled_y = sampled_y.cpu()
        train_x = train_x.cpu()
        kernel_eval = kernel_eval.cpu()
        posterior_sample = posterior_sample.cpu()
        mean = mean.cpu()
        # std = std.cpu()

    mlab.figure()
    mlab.triangular_mesh(nodes[:, 0], nodes[:, 1],
                         nodes[:, 2], faces, scalars=mean.numpy())

    mlab.triangular_mesh(nodes[:, 0], nodes[:, 1],
                         nodes[:, 2], faces, scalars=sampled_y.numpy())

    # # Sampled Points
    # ax = fig.add_subplot(231, projection='3d')
    # ax.scatter(sampled_x[perm_idx[num_train:], 0],
    #            sampled_x[perm_idx[num_train:], 1], sampled_x[perm_idx[num_train:], 2])
    # ax.scatter(train_x[:, 0], train_x[:, 1],
    #            train_x[:, 2], c="r", edgecolors="r", s=20)
    # ax.set_title('Training Points')

    # # Ground Truth
    # ax = fig.add_subplot(232, projection='3d')
    # plot = ax.scatter(sampled_x[:, 0], sampled_x[:, 1],
    #                   sampled_x[:, 2], c=sampled_y)
    # fig.colorbar(plot)
    # ax.set_title('Ground Truth')

    # # Kernel evaluation
    # ax = fig.add_subplot(233, projection='3d')
    # plot = ax.scatter(sampled_x[:, 0], sampled_x[:, 1],
    #                   sampled_x[:, 2], c=kernel_eval)
    # ax.scatter(sampled_x[0, 0], sampled_x[0, 1], c='k')
    # fig.colorbar(plot)
    # ax.set_title('Kernel Evaluation')

    # # One Posterior Sample
    # ax = fig.add_subplot(234, projection='3d')
    # plot = ax.scatter(sampled_x[:, 0], sampled_x[:, 1],
    #                   sampled_x[:, 2], c=posterior_sample)
    # fig.colorbar(plot)
    # ax.set_title('One Posterior Sample')

    # # Mean
    # ax = fig.add_subplot(235, projection='3d')
    # plot = ax.scatter(sampled_x[:, 0], sampled_x[:,
    #                   1], sampled_x[:, 2], c=mean)
    # fig.colorbar(plot)
    # ax.set_title('Mean')

    # # Standard Deviation
    # ax = fig.add_subplot(236)
    # plot = ax.scatter(sampled_x[:, 0], sampled_x[:, 1], c=std)
    # fig.colorbar(plot)
    # ax.axis('equal')
    # ax.set_title('Standard Deviation')

plt.show()
