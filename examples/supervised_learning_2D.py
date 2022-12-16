#!/usr/bin/env python
# encoding: utf-8

# import numpy as np
# import torch
# import gpytorch
# import matplotlib.pyplot as plt
# from importlib.resources import files
# from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
# from manifold_gp.models.riemann_gp import RiemannGP
# from manifold_gp.utils.generate_truth import groundtruth_from_mesh

# # Set device
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")

# # Load mesh and generate ground truth
# data_path = files('manifold_gp.data').joinpath('dragon10k.stl')
# nodes, faces, truth = groundtruth_from_mesh(data_path)

# # Generate ground truth
# sampled_x = torch.from_numpy(nodes).float().to(device)
# sampled_y = torch.from_numpy(truth).float().to(device)
# (m, n) = sampled_x.shape

# # Noisy dataset
# manifold_noise = 0.0
# noisy_x = sampled_x + manifold_noise * torch.randn(m, n).to(device)
# function_noise = 0.01
# noisy_y = sampled_y + function_noise * torch.randn(m).to(device)

# # Train dataset
# num_train = 500
# train_idx = torch.randperm(m)[:num_train]
# train_x = noisy_x[train_idx, :]
# train_y = noisy_y[train_idx]

# # Test points
# num_test = 100
# test_idx = torch.randperm(m)[:num_test]
# test_x = noisy_x[test_idx, :]
# test_y = noisy_y[test_idx]

# # Initialize kernel
# nu = 2
# neighbors = 50
# modes = 20
# kernel = gpytorch.kernels.ScaleKernel(
#     RiemannMaternKernel(sampled_x, nu, neighbors, modes))

# # kernel2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())


# # class ExactGPModel(gpytorch.models.ExactGP):
# #     def __init__(self, train_x, train_y, likelihood):
# #         super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
# #         self.mean_module = gpytorch.means.ConstantMean()
# #         self.covar_module = gpytorch.kernels.ScaleKernel(
# #             gpytorch.kernels.RBFKernel())

# #     def forward(self, x):
# #         mean_x = self.mean_module(x)
# #         covar_x = self.covar_module(x)
# #         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# # Initialize likelihood and model
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
# # model2 = ExactGPModel(sampled_x, sampled_y, likelihood).to(device)
# model = RiemannGP(sampled_x, sampled_y, likelihood, kernel).to(device)

# # Train model
# lr = 1e-1
# iters = 10
# verbose = True
# model.manifold_informed_train(lr, iters, verbose)

# # Model Evaluation
# likelihood.eval()
# model.eval()

# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     preds = likelihood(model(sampled_x))

#     # Mean
#     mean = preds.mean

#     # Standard deviation
#     std = preds.stddev

#     # One Posterior Sample
#     posterior_sample = preds.sample()

#     # Kernel evaluation
#     kernel_eval = kernel(sampled_x[0, :].unsqueeze(
#         0), sampled_x).evaluate().squeeze()

# with torch.no_grad():
#     # Bring data to cpu
#     if use_cuda:
#         sampled_x = sampled_x.cpu()
#         sampled_y = sampled_y.cpu()
#         train_x = train_x.cpu()
#         kernel_eval = kernel_eval.cpu()
#         posterior_sample = posterior_sample.cpu()
#         mean = mean.cpu()
#         std = std.cpu()

#     fig = plt.figure(figsize=(17, 9))

#     # Sampled Points
#     ax = fig.add_subplot(231, projection='3d')
#     ax.scatter(sampled_x[:, 0], sampled_x[:, 1], sampled_x[:, 2])
#     ax.scatter(train_x[:, 0], train_x[:, 1],
#                train_x[:, 2], c="r", edgecolors="r", s=20)
#     ax.set_title('Training Points')

#     # Ground Truth
#     ax = fig.add_subplot(232, projection='3d')
#     plot = ax.scatter(sampled_x[:, 0], sampled_x[:, 1],
#                       sampled_x[:, 2], c=sampled_y)
#     fig.colorbar(plot)
#     ax.set_title('Ground Truth')

#     # Kernel evaluation
#     ax = fig.add_subplot(233, projection='3d')
#     plot = ax.scatter(sampled_x[:, 0], sampled_x[:, 1],
#                       sampled_x[:, 2], c=kernel_eval)
#     ax.scatter(sampled_x[0, 0], sampled_x[0, 1], c='k')
#     fig.colorbar(plot)
#     ax.set_title('Kernel Evaluation')

#     # One Posterior Sample
#     ax = fig.add_subplot(234, projection='3d')
#     plot = ax.scatter(sampled_x[:, 0], sampled_x[:, 1],
#                       sampled_x[:, 2], c=posterior_sample)
#     fig.colorbar(plot)
#     ax.set_title('One Posterior Sample')

#     # Mean
#     ax = fig.add_subplot(235, projection='3d')
#     plot = ax.scatter(sampled_x[:, 0], sampled_x[:,
#                       1], sampled_x[:, 2], c=mean)
#     fig.colorbar(plot)
#     ax.set_title('Mean')

#     # Standard Deviation
#     ax = fig.add_subplot(236, projection='3d')
#     plot = ax.scatter(sampled_x[:, 0], sampled_x[:, 1], sampled_x[:, 2], c=std)
#     fig.colorbar(plot)
#     ax.set_title('Standard Deviation')

# plt.show()

from mayavi import mlab
from numpy import sin, cos, mgrid, pi, sqrt
mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
u, v = mgrid[- 0.035:pi:0.01, - 0.035:pi:0.01]

X = 2 / 3. * (cos(u) * cos(2 * v)
              + sqrt(2) * sin(u) * cos(v)) * cos(u) / (sqrt(2) -
                                                       sin(2 * u) * sin(3 * v))
Y = 2 / 3. * (cos(u) * sin(2 * v) -
              sqrt(2) * sin(u) * sin(v)) * cos(u) / (sqrt(2)
                                                     - sin(2 * u) * sin(3 * v))
Z = -sqrt(2) * cos(u) * cos(u) / (sqrt(2) - sin(2 * u) * sin(3 * v))

for i in range(1, 10):
    S = sin(u)*i
    mlab.mesh(X, Y, Z, scalars=S, colormap='YlGnBu', )
    mlab.view(.0, - 5.0, 4)
    mlab.colorbar(orientation='vertical')
    mlab.show()
    mlab.close(all=True)
