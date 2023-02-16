# !/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import gpytorch

from mayavi import mlab
from importlib.resources import files

from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.models.riemann_gp import RiemannGP

from manifold_gp.utils.generate_truth import groundtruth_from_mesh


# Set device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load mesh and generate ground truth
data_path = files('manifold_gp.data').joinpath('dragon10k.stl')
nodes, faces, truth = groundtruth_from_mesh(data_path)
sampled_x = torch.from_numpy(nodes).float().to(device)
sampled_y = torch.from_numpy(truth).float().to(device)
(m, n) = sampled_x.shape

# Normalization
# mu, std = sampled_x.mean(0), sampled_x.std(0)
# sampled_x.sub_(mu).div_(std)

# Noisy dataset
manifold_noise = 0.0
noisy_x = sampled_x + manifold_noise * torch.randn(m, n).to(device)
function_noise = 0.0
noisy_y = sampled_y + function_noise * torch.randn(m).to(device)

# Train dataset
num_train = 100
train_idx = torch.randperm(m)[:num_train]
train_x = noisy_x[train_idx, :]
train_y = noisy_y[train_idx]

# Test dataset
num_test = 100
test_idx = torch.randperm(m)[:num_test]
test_x = noisy_x[test_idx, :]
test_y = noisy_y[test_idx]

# Initialize kernel
nu = 2
neighbors = 10
modes = 50
kernel = gpytorch.kernels.ScaleKernel(RiemannMaternKernel(nu=nu, nodes=train_x, neighbors=neighbors, modes=modes))

# Initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-8))
model = RiemannGP(train_x, train_y, likelihood, kernel).to(device)

# Model Hyperparameters
hypers = {
    'likelihood.noise_covar.noise': 1e-5,
    'covar_module.base_kernel.epsilon': 0.5,
    'covar_module.base_kernel.lengthscale': 0.5,
    'covar_module.outputscale': 1.0,
}
model.initialize(**hypers)

# Train model
lr = 1e-2
iters = 100
verbose = True
loss = model.manifold_informed_train(lr, iters, verbose)

# Model Evaluation
likelihood.eval()
model.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # GP posteriors
    preds = likelihood(model(noisy_x))
    mean = preds.mean.cpu().numpy()
    std = preds.stddev.cpu().numpy()
    posterior_sample = preds.sample().cpu().numpy()

    # Prior Variance
    var = kernel.base_kernel.variance(noisy_x).cpu().numpy()

    # Kernel evaluation
    point = 3405
    kernel_eval = kernel(noisy_x[point, :].unsqueeze(0), noisy_x).evaluate().squeeze().cpu().numpy()

    # Eigenfunctions
    mode = 1
    eigfunctions = kernel.base_kernel.eigenfunctions(noisy_x).cpu().numpy()

    # Bring data to numpy
    sampled_x = sampled_x.cpu().numpy()
    sampled_y = sampled_y.cpu().numpy()
    train_x = train_x.cpu().numpy()

v_options = {'mode': 'sphere', 'scale_factor': 3e-3, 'color': (0, 0, 0)}
# Ground Truth
mlab.figure('Groud Truth', size=(1920, 1360), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1], sampled_x[:, 2], faces, scalars=sampled_y)
mlab.colorbar(orientation='vertical')
mlab.view(0.0, 180.0, 0.5139171204775793)
mlab.points3d(train_x[:, 0], train_x[:, 1], train_x[:, 2], **v_options)

# Mean
mlab.figure('Mean', size=(1920, 1360), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1], sampled_x[:, 2], faces, scalars=mean)
mlab.colorbar(orientation='vertical')
mlab.view(0.0, 180.0, 0.5139171204775793)
mlab.points3d(train_x[:, 0], train_x[:, 1], train_x[:, 2], **v_options)

# Standard Deviation
mlab.figure('Standard Deviation', size=(1920, 1360), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1], sampled_x[:, 2], faces, scalars=std)
mlab.colorbar(orientation='vertical')
mlab.view(0.0, 180.0, 0.5139171204775793)
mlab.points3d(train_x[:, 0], train_x[:, 1], train_x[:, 2], **v_options)

# One Posterior Sample
mlab.figure('One Posterior Sample', size=(1920, 1360), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1], sampled_x[:, 2], faces, scalars=posterior_sample)
mlab.colorbar(orientation='vertical')
mlab.view(0.0, 180.0, 0.5139171204775793)
mlab.points3d(train_x[:, 0], train_x[:, 1], train_x[:, 2], **v_options)

# Prior Variance
mlab.figure('Prior Variance', size=(1920, 1360), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1], sampled_x[:, 2], faces, scalars=var)
mlab.colorbar(orientation='vertical')
mlab.view(0.0, 180.0, 0.5139171204775793)
mlab.points3d(train_x[:, 0], train_x[:, 1], train_x[:, 2], **v_options)

# Kernel evaluation
kernel_eval = kernel_eval - np.min(kernel_eval)
kernel_eval /= np.max(kernel_eval)
mlab.figure('Kernel Evaluation', size=(1920, 1360), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1], sampled_x[:, 2], faces, scalars=kernel_eval)
mlab.colorbar(orientation='vertical')
mlab.points3d(sampled_x[point, 0], sampled_x[point, 1], sampled_x[point, 2], **v_options)
mlab.view(0.0, 180.0, 0.5139171204775793)

# Eigenfunction
eigfun = eigfunctions[mode, :] - np.min(eigfunctions[mode, :])
eigfun /= np.max(eigfun)
mlab.figure('Eigenfunction', size=(1920, 1360), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1], sampled_x[:, 2], faces, scalars=eigfun)
mlab.colorbar(orientation='vertical')
mlab.view(0.0, 180.0, 0.5139171204775793)
