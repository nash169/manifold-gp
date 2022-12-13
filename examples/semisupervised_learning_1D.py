#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import gpytorch
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.models.riemann_gp import RiemannGP

# Set device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load dataset
data = np.loadtxt('rsc/dumbbell.msh')
sampled_x = torch.from_numpy(data[:, :2]).float().to(device)
sampled_y = torch.from_numpy(data[:, -1][:, np.newaxis]).float().to(device)
(m, n) = sampled_x.shape

# Noisy dataset
manifold_noise = 0.01
function_noise = 0.01
sampled_x = sampled_x + manifold_noise * torch.randn(m, n).to(device)
sampled_y = sampled_y + function_noise * torch.randn(m, 1).to(device)

# Train dataset
num_train = 100
train_idx = torch.randint(m, (num_train,))
train_x = sampled_x[train_idx, :]
train_y = sampled_y[train_idx, :]

# Test points
num_test = 50
test_idx = torch.randint(m, (num_test,))
test_x = sampled_x[test_idx, :]
test_y = sampled_y[test_idx]

# Initialize kernel
nu = 2
neighbors = 5
modes = 10
kernel = gpytorch.kernels.ScaleKernel(
    RiemannMaternKernel(sampled_x, nu, neighbors, modes))

# Initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = RiemannGP(train_x, train_y, likelihood, kernel, train_idx).to(device)

# Train model
lr = 1e-1
iters = 100
verbose = True
model.manifold_informed_train(lr, iters, verbose)
