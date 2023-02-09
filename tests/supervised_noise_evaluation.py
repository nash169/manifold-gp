#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from importlib.resources import files
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.models.riemann_gp import RiemannGP

# Set device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load dataset
data_path = files('manifold_gp.data').joinpath('dumbbell.msh')
data = np.loadtxt(data_path)
sampled_x = torch.from_numpy(data[:, :2]).float().to(device)
sampled_y = torch.from_numpy(data[:, -1]).float().to(device)
(m, n) = sampled_x.shape

# Set Train Indices
num_train = 50
train_idx = torch.randperm(m)[:num_train]

# Set Test Indices
num_test = 10
test_idx = torch.randperm(m)[:num_test]

# Noise Grid
min_noise_scale = 0
max_noise_scale = 0.05
resolution = 3
manifold_noise, function_noise = torch.meshgrid(torch.linspace(min_noise_scale, max_noise_scale, steps=resolution),
                                                torch.linspace(min_noise_scale, max_noise_scale, steps=resolution))

# Record Loss & MSE
count = 1
loss = torch.zeros_like(manifold_noise)
mse = torch.zeros_like(manifold_noise)

# Create Kernel
nu = 2
neighbors = 50
modes = 10
kernel = gpytorch.kernels.ScaleKernel(
    RiemannMaternKernel(sampled_x, nu, neighbors, modes))

# Define Optimization Parameters
lr = 1e-1
iters = 50
verbose = False

# Model Hyperparameters
hypers = {
    'likelihood.noise_covar.noise': torch.tensor(0.007),
    'covar_module.base_kernel.epsilon': torch.tensor(0.13),
    'covar_module.base_kernel.lengthscale': torch.tensor(0.37),
    'covar_module.outputscale': torch.tensor(1.),
}

for i in range(resolution):
    for j in range(resolution):
        print(
            f"Iteration: {count}/{resolution**2}, Manifold Noise: {manifold_noise[i,j]}, Function Noise: {function_noise[i,j]}")
        # Add noise to samples
        noisy_x = sampled_x + \
            manifold_noise[i, j] * torch.randn(m, n).to(device)
        noisy_y = sampled_y + function_noise[i, j] * torch.randn(m).to(device)

        # Train dataset
        train_x = noisy_x[train_idx, :]
        train_y = noisy_y[train_idx]

        # Test dataset
        test_x = noisy_x[test_idx, :]
        test_y = noisy_y[test_idx]

        # Update Kernel
        if hasattr(kernel, 'base_kernel'):
            kernel.base_kernel.nodes = noisy_x
            kernel.base_kernel.generate_graph()
        else:
            kernel.nodes = noisy_x
            kernel.generate_graph()

        # Create model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = RiemannGP(noisy_x, noisy_y, likelihood, kernel).to(device)

        # Init hyperparameters and train the model
        model.initialize(**hypers)
        loss[i, j] = model.manifold_informed_train(lr, iters, verbose)

        # Evaluate the model
        likelihood.eval()
        model.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = likelihood(model(noisy_x))
            mse[i, j] = torch.linalg.norm(sampled_y - preds.mean)

        count += 1

with torch.no_grad():
    fig = plt.figure()
    ax = fig.add_subplot(121)
    contour_mse = ax.contourf(function_noise, manifold_noise, mse)
    fig.colorbar(contour_mse)

    ax = fig.add_subplot(122)
    contour_loss = ax.contourf(function_noise, manifold_noise, loss)
    fig.colorbar(contour_loss)
