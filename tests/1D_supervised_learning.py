#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import gpytorch

import matplotlib.pyplot as plt
from importlib.resources import files

from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.models.riemann_gp import RiemannGP
from manifold_gp.models.vanilla_gp import VanillaGP
from manifold_gp.utils.generate_truth import groundtruth_from_samples

data_path = files('manifold_gp.data').joinpath('dumbbell240.msh')
data = np.loadtxt(data_path)
sampled_x = torch.from_numpy(data[:, :2]).float()
sampled_y = torch.from_numpy(data[:, -1]).float()

# Load mesh and generate ground truth
# data_path = files('manifold_gp.data').joinpath('dumbbell.msh')
# nodes = np.loadtxt(data_path)
# truth = groundtruth_from_samples(nodes)
# sampled_x = torch.from_numpy(nodes).float()
# sampled_y = torch.from_numpy(truth).float()
(m, n) = sampled_x.shape

# features/labels noise
manifold_noise = 0.0
noisy_x = sampled_x + manifold_noise * torch.randn(m, n)
function_noise = 0.0
noisy_y = sampled_y + function_noise * torch.randn(m)

# features/labels normalization
# mu_f, std_f = noisy_x.mean(dim=-2, keepdim=True), noisy_x.std(dim=-2, keepdim=True) + 1e-6
# noisy_x.sub_(mu_f).div_(std_f)
mu_l, std_l = noisy_y.mean(), noisy_y.std()
noisy_y.sub_(mu_l).div_(std_l)

# train/test set
torch.manual_seed(1337)
rand_idx = torch.randperm(m)
num_train = 50
train_idx = rand_idx[:num_train]
train_x, train_y = noisy_x[train_idx, :], noisy_y[train_idx]
test_idx = rand_idx[num_train:]
test_x, test_y = noisy_x[test_idx, :], noisy_y[test_idx]

# make contiguous
noisy_x, noisy_y = noisy_x.contiguous(), noisy_y.contiguous()
train_x, train_y = train_x.contiguous(), train_y.contiguous()
test_x, test_y = test_x.contiguous(), test_y.contiguous()

# bring to device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
noisy_x, noisy_y = noisy_x.to(device), noisy_y.to(device)
train_x, train_y = train_x.to(device), train_y.to(device)
test_x, test_y = test_x.to(device), test_y.to(device)

# kernel
kernel = gpytorch.kernels.ScaleKernel(
    RiemannMaternKernel(
        nu=2,
        nodes=train_x,
        neighbors=20,  # int(num_train*0.25),
        modes=40,  # int(num_train*0.5),
        support_kernel=gpytorch.kernels.RBFKernel(),
        epsilon_prior=None,  # GammaPrior(gamma_concentration, gamma_rate),
        lengthscale_prior=None  # InverseGammaPrior(igamma_concentration, igamma_rate)
    ),
    outputscale_prior=None  # NormalPrior(torch.tensor([1.0]).to(device),  torch.tensor([1/9]).sqrt().to(device))
)

# likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
    noise_prior=None  # NormalPrior(torch.tensor([0.0]).to(device),  torch.tensor([1/9]).sqrt().to(device))
)

# model and train
# model = RiemannGP(train_x, train_y, likelihood, kernel).to(device)
# hypers = {
#     'likelihood.noise_covar.noise': 1e-5,
#     'covar_module.base_kernel.epsilon': 0.5,
#     'covar_module.base_kernel.lengthscale': 0.5,
#     'covar_module.outputscale': 1.0,
#     'covar_module.base_kernel.support_kernel.lengthscale': 1.0,
# }
# model.initialize(**hypers)
# model.manifold_informed_train(lr=1e-2, iter=100, verbose=True)

model = VanillaGP(noisy_x, noisy_y, likelihood, gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))).to(device)
# gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
# gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
hypers = {
    'likelihood.noise_covar.noise': 1e-5,
    'covar_module.base_kernel.lengthscale': 0.5,
    'covar_module.outputscale': 1.0,
}
model.initialize(**hypers)
model.vanilla_train(lr=1e-2, iter=100, verbose=True)

# evaluation
likelihood.eval()
model.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # manifold
    preds = likelihood(model(noisy_x))
    posterior_mean = preds.mean.mul(std_l).add(mu_l).cpu().numpy()
    posterior_std = preds.stddev.mul(std_l).add(mu_l).cpu().numpy()
    posterior_sample = preds.sample().mul(std_l).add(mu_l).cpu().numpy()
    # prior_var = model.covar_module.variance(noisy_x).cpu().numpy()
    prior_var = model.covar_module(noisy_x, noisy_x).evaluate().diag().cpu().numpy()
    kernel_eval = model.covar_module(noisy_x[0, :].unsqueeze(0), noisy_x).evaluate().squeeze().cpu().numpy()
    # eigfunction = kernel.base_kernel.eigenvectors[:, 1].cpu().numpy()

    # ambient
    resolution = 100
    y, x = torch.meshgrid(torch.linspace(-1.5, 1.5, resolution), torch.linspace(-1.5, 1.5, resolution))
    grid_x = torch.stack((torch.ravel(x), torch.ravel(y)), dim=1).to(device).requires_grad_(True)
    preds = likelihood(model(grid_x))
    posterior_mean_grid = preds.mean.mul(std_l).add(mu_l).cpu().numpy()
    posterior_std_grid = preds.stddev.mul(std_l).add(mu_l).cpu().numpy()
    posterior_sample_grid = preds.sample().mul(std_l).add(mu_l).cpu().numpy()
    prior_var_grid = model.covar_module(grid_x, grid_x).evaluate().diag().cpu().numpy()
    # prior_var_grid = model.covar_module.variance(grid_x).cpu().numpy()
    kernel_eval_grid = model.covar_module(noisy_x[0, :].unsqueeze(0), grid_x).evaluate().squeeze().cpu().numpy()
    # eigfunctions_grid = kernel.base_kernel.eigenfunctions(grid_x)[1, :].cpu().numpy()

    # metrics
    preds = likelihood(model(test_x))
    rmse = (preds.mean - test_y).square().sum().sqrt()
    nll = -preds.log_prob(test_y)

    # Bring data to numpy
    sampled_x = sampled_x.numpy()
    sampled_y = sampled_y.numpy()
    train_x = train_x.cpu().numpy()
    train_y = train_y.cpu().numpy()
    # noisy_x = noisy_x.cpu().mul(std_f).add(mu_f).numpy()
    noisy_x = noisy_x.cpu().numpy()

fig = plt.figure(figsize=(17, 9))

# Ground Truth
ax = fig.add_subplot(231)
plot = ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c=sampled_y)  # vmin=-0.5, vmax=0.5
fig.colorbar(plot)
ax.axis('equal')
ax.set_title('Ground Truth')

# Posterior Mean
ax = fig.add_subplot(232)
plot = ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c=posterior_mean)
ax.scatter(train_x[:, 0], train_x[:, 1], c='k', s=0.5)
fig.colorbar(plot)
ax.axis('equal')
ax.set_title('Posterior Mean')

# Kernel evaluation
ax = fig.add_subplot(233)
plot = ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c=kernel_eval)
ax.scatter(noisy_x[0, 0], noisy_x[0, 1], c='k', s=0.5)
fig.colorbar(plot)
ax.axis('equal')
ax.set_title('Kernel Evaluation')

# # Posterior Sample
# ax = fig.add_subplot(234)
# plot = ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c=posterior_sample)
# ax.scatter(train_x[:, 0], train_x[:, 1], c='k', s=0.5)
# fig.colorbar(plot)
# ax.axis('equal')
# ax.set_title('Posterior Sample')

# Standard Deviation
ax = fig.add_subplot(235)
plot = ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c=posterior_std)
ax.scatter(train_x[:, 0], train_x[:, 1], c='k', s=0.5)
fig.colorbar(plot)
ax.axis('equal')
ax.set_title('Standard Deviation')

# Prior variance
ax = fig.add_subplot(236)
plot = ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c=prior_var)
ax.scatter(train_x[:, 0], train_x[:, 1], c='k', s=0.5)
fig.colorbar(plot)
ax.axis('equal')
ax.set_title('Prior Variance')


fig = plt.figure(figsize=(17, 9))

# Ground Truth
ax = fig.add_subplot(231)
plot = ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c=sampled_y)  # vmin=-0.5, vmax=0.5
fig.colorbar(plot)
ax.axis('equal')
ax.set_title('Ground Truth')

# Posterior Mean
ax = fig.add_subplot(232)
contour = ax.contourf(x.cpu(), y.cpu(), posterior_mean_grid.reshape(resolution, -1), 500)
# ax.contour(x.cpu(), y.cpu(), posterior_mean_grid.reshape(resolution, -1), 10, cmap=None, colors='#f2e68f')
ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c='k', s=0.5)
ax.scatter(train_x[:, 0], train_x[:, 1], c='r', s=0.5)
fig.colorbar(contour)
ax.axis('square')
ax.set_title('Posterior Mean')

# Kernel evaluation
ax = fig.add_subplot(233)
contour = ax.contourf(x.cpu(), y.cpu(), kernel_eval_grid.reshape(resolution, -1), 500)
# ax.contour(x.cpu(), y.cpu(), kernel_eval_grid.reshape(resolution, -1), 10, cmap=None, colors='#f2e68f')
ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c='k', s=0.5)
ax.scatter(noisy_x[0, 0], noisy_x[0, 1], c='k', s=1.0)
fig.colorbar(contour)
ax.axis('square')
ax.set_title('Kernel Evaluation')

# Posterior Sample
ax = fig.add_subplot(234)
contour = ax.contourf(x.cpu(), y.cpu(), posterior_sample_grid.reshape(resolution, -1), 500)
# ax.contour(x.cpu(), y.cpu(), posterior_sample_grid.reshape(resolution, -1), 10, cmap=None, colors='#f2e68f')
ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c='k', s=0.5)
ax.scatter(train_x[:, 0], train_x[:, 1], c='r', s=0.5)
fig.colorbar(contour)
ax.axis('square')
ax.set_title('Posterior Sample')

# Standard Deviation
ax = fig.add_subplot(235)
contour = ax.contourf(x.cpu(), y.cpu(), posterior_std_grid.reshape(resolution, -1), 500)
# ax.contour(x.cpu(), y.cpu(), posterior_std_grid.reshape(resolution, -1), 10, cmap=None, colors='#f2e68f')
ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c='k', s=0.5)
ax.scatter(train_x[:, 0], train_x[:, 1], c='r', s=0.5)
fig.colorbar(contour)
ax.axis('square')
ax.set_title('Standard Deviation')

# Prior variance
ax = fig.add_subplot(236)
contour = ax.contourf(x.cpu(), y.cpu(), prior_var_grid.reshape(resolution, -1), 500)
# ax.contour(x.cpu(), y.cpu(), prior_var_grid.reshape(resolution, -1), 10, cmap=None, colors='#f2e68f')
ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c='k', s=0.5)
ax.scatter(train_x[:, 0], train_x[:, 1], c='r', s=0.5)
fig.colorbar(contour)
ax.axis('square')
ax.set_title('Prior Variance')

fig = plt.figure()
ax = fig.add_subplot(111)
plot = ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c=posterior_std)
ax.scatter(train_x[:, 0], train_x[:, 1], c='r', s=0.7)
fig.colorbar(plot)
ax.axis('equal')
ax.set_title('Standard Deviation')

# plt.show(block=False)
plt.show()
