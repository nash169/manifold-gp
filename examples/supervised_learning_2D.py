# !/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from mayavi import mlab
from importlib.resources import files
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.kernels.riemann_rbf_kernel import RiemannRBFKernel
from manifold_gp.models.riemann_gp import RiemannGP
from manifold_gp.utils.generate_truth import groundtruth_from_mesh

# Set device
use_cuda = False  # torch.cuda.is_available()
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
function_noise = 0.0
noisy_y = sampled_y + function_noise * torch.randn(m).to(device)

# Train dataset
num_train = 500
train_idx = torch.randperm(m)[:num_train]
train_x = noisy_x[train_idx, :]
train_y = noisy_y[train_idx]

# Test points
num_test = 100
test_idx = torch.randperm(m)[:num_test]
test_x = noisy_x[test_idx, :]
test_y = noisy_y[test_idx]

# Initialize kernel
nu = 2
neighbors = 10
modes = 100
# kernel = gpytorch.kernels.ScaleKernel(
#     RiemannMaternKernel(nu=nu, nodes=noisy_x, neighbors=neighbors, modes=modes))
kernel = gpytorch.kernels.ScaleKernel(
    RiemannRBFKernel(nodes=noisy_x, neighbors=neighbors, modes=modes))

# Initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-8))
model = RiemannGP(noisy_x, noisy_y, likelihood, kernel).to(device)

# Model Hyperparameters
hypers = {
    'likelihood.noise_covar.noise': 1e-3,
    'covar_module.base_kernel.epsilon': 0.5027,
    'covar_module.base_kernel.lengthscale': 0.5054,
    'covar_module.outputscale': 1,
}
model.initialize(**hypers)

# Train model
# lr = 1e-1
# iters = 20
# verbose = True
# model.manifold_informed_train(lr, iters, verbose)

# Model Evaluation
likelihood.eval()
model.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds = likelihood(model(sampled_x))

    # Mean
    mean = preds.mean

    # Standard deviation
    std = preds.stddev

    # One Posterior Sample
    posterior_sample = preds.sample()

    # Kernel evaluation
    kernel_eval = kernel(sampled_x[0, :].unsqueeze(
        0), sampled_x).evaluate().squeeze()

    # Bring data to cpu
    sampled_x = sampled_x.cpu().numpy()
    sampled_y = sampled_y.cpu().numpy()
    train_x = train_x.cpu().numpy()
    kernel_eval = kernel_eval.cpu().numpy()
    posterior_sample = posterior_sample.cpu().numpy()
    mean = mean.cpu().numpy()
    std = std.cpu().numpy()

v_options = {'mode': 'sphere', 'scale_factor': 3e-3, 'color': (0, 0, 0)}

# Ground Truth
mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1],
                     sampled_x[:, 2], faces, scalars=sampled_y)
mlab.colorbar(orientation='vertical')

# Mean
mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1],
                     sampled_x[:, 2], faces, scalars=mean)
mlab.colorbar(orientation='vertical')

# # # Standard Deviation
# # mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# # mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1],
# #                      sampled_x[:, 2], faces, scalars=std)
# # mlab.colorbar(orientation='vertical')

# # # One Posterior Sample
# # mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# # mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1],
# #                      sampled_x[:, 2], faces, scalars=posterior_sample)
# # mlab.colorbar(orientation='vertical')

# # # Kernel evaluation
# # mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# # mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1],
# #                      sampled_x[:, 2], faces, scalars=kernel_eval)
# # mlab.colorbar(orientation='vertical')
