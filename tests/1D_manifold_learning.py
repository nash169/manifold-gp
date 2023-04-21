# !/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import gpytorch

import matplotlib.pyplot as plt
from importlib.resources import files

from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.models.riemann_gp import RiemannGP
from manifold_gp.models.vanilla_gp import VanillaGP

from manifold_gp.utils.file_read import get_data
from manifold_gp.utils.mesh_helper import groundtruth_from_samples, plot_1D

import networkx as nx

from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

# Load mesh and generate ground truth
data_path = files('manifold_gp.data').joinpath('dumbbell.msh')
data = get_data(data_path, "Nodes", "Elements")

vertices = data['Nodes'][:, 1:-1]
edges = data['Elements'][:, -2:].astype(int) - 1
truth = groundtruth_from_samples(vertices)

# # # vertices = [(0, 0), (1, 1), (1, 2), (0, 2)]
# # # edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
# vertices = [tuple(vertex) for vertex in vertices]
# edges = [tuple(edge) for edge in edges]

# G = nx.Graph()
# G.add_nodes_from(range(len(vertices)))
# G.add_edges_from(edges)

# nx.draw(G, pos=dict(enumerate(vertices)), with_labels=True)
# vertices.append(vertices[0])
# plt.plot(*zip(*vertices))
# plt.show(block=False)

# z = [0, 1]


fig, ax = plt.subplots()
plot_1D(fig, ax, vertices, edges, truth)
# tmp = vertices.reshape(-1, 1, 2)
# segments = np.concatenate([tmp[edges[:, 0]], tmp[edges[:, 1]]], axis=1)
# norm = plt.Normalize(truth.min(), truth.max())
# lc = LineCollection(segments, cmap='viridis', norm=norm)
# lc.set_array(truth)
# lc.set_linewidth(2)
# line = ax.add_collection(lc)
# fig.colorbar(line, ax=ax)
# ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
# ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
# ax.axis('equal')


# for edge in edges:
#     start = vertices[edge[0]]
#     end = vertices[edge[1]]
#     # ax.plot([start[0], end[0]], [start[1], end[1]])
#     # lc = LineCollection([[[start[0], end[0]], [start[1], end[1]]]], cmap='viridis', norm=norm)
#     # lc.set_array(truth)
#     # lc.set_linewidth(2)
#     # line = ax.add_collection(lc)
#     # fig.colorbar(line, ax=ax)
#     # break

fig = plt.figure()
# Ground Truth
ax = fig.add_subplot(111)
plot = ax.scatter(vertices[:, 0], vertices[:, 1], c=truth)  # vmin=-0.5, vmax=0.5
fig.colorbar(plot)
ax.axis('equal')

plt.show()

# # Create a figure and axes
# fig, ax = plt.subplots()

# # Define the endpoints of the segment and their scalar values
# x = [0, 1]
# y = [0, 1]
# z = [0, 1]

# # Create a color map
# cmap = plt.cm.get_cmap('RdBu')

# # Plot the segment with a color gradient
# ax.plot(x, y, c=z, cmap=cmap)

# fig = plt.figure()
# # Ground Truth
# ax = fig.add_subplot(111)

# x = np.stack((nodes[elements[:, 0], 0], nodes[elements[:, 1], 0]), axis=1).reshape(-1, 1, order='C')
# y = np.stack((nodes[elements[:, 0], 1], nodes[elements[:, 1], 1]), axis=1).reshape(-1, 1, order='C')
# # plot = ax.scatter(nodes[:, 0], nodes[:, 1])  # vmin=-0.5, vmax=0.5
# plt.plot(x, y)
# # fig.colorbar(plot)
# ax.axis('equal')
# plt.show()

# sampled_x = torch.from_numpy(data[:, :2]).float()
# sampled_y = torch.from_numpy(data[:, -1]).float()
# (m, n) = sampled_x.shape

# # features/labels noise
# manifold_noise = 0.0
# noisy_x = sampled_x + manifold_noise * torch.randn(m, n)
# function_noise = 0.0
# noisy_y = sampled_y + function_noise * torch.randn(m)

# # features/labels normalization
# # mu_f, std_f = noisy_x.mean(dim=-2, keepdim=True), noisy_x.std(dim=-2, keepdim=True) + 1e-6
# # noisy_x.sub_(mu_f).div_(std_f)
# mu_l, std_l = noisy_y.mean(), noisy_y.std()
# noisy_y.sub_(mu_l).div_(std_l)

# # make contiguous
# noisy_x, noisy_y = noisy_x.contiguous(), noisy_y.contiguous()

# # bring to device
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
# noisy_x, noisy_y = noisy_x.to(device), noisy_y.to(device)

# # kernel
# kernel = gpytorch.kernels.ScaleKernel(
#     RiemannMaternKernel(
#         nu=2,
#         nodes=noisy_x,
#         neighbors=int(m*0.25),
#         modes=int(m*0.5),
#         support_kernel=gpytorch.kernels.RBFKernel(),
#         epsilon_prior=None,  # GammaPrior(gamma_concentration, gamma_rate),
#         lengthscale_prior=None  # InverseGammaPrior(igamma_concentration, igamma_rate)
#     ),
#     outputscale_prior=None  # NormalPrior(torch.tensor([1.0]).to(device),  torch.tensor([1/9]).sqrt().to(device))
# )

# # likelihood
# likelihood = gpytorch.likelihoods.GaussianLikelihood(
#     noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
#     noise_prior=None  # NormalPrior(torch.tensor([0.0]).to(device),  torch.tensor([1/9]).sqrt().to(device))
# )

# # model and train
# model = RiemannGP(noisy_x, noisy_y, likelihood, kernel).to(device)
# hypers = {
#     'likelihood.noise_covar.noise': 1e-5,
#     'covar_module.base_kernel.epsilon': 0.5,
#     'covar_module.base_kernel.lengthscale': 0.5,
#     'covar_module.outputscale': 1.0,
#     'covar_module.base_kernel.support_kernel.lengthscale': 0.1,
# }
# model.initialize(**hypers)
# model.manifold_informed_train(lr=1e-2, iter=100, verbose=True)

# # # gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
# # model = VanillaGP(noisy_x, noisy_y, likelihood, gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())).to(device)
# # hypers = {
# #     'likelihood.noise_covar.noise': 1e-5,
# #     'covar_module.base_kernel.lengthscale': 0.5,
# #     'covar_module.outputscale': 1.0,
# # }
# # model.initialize(**hypers)
# # model.train(lr=1e-2, iter=100, verbose=True)

# # evaluation
# likelihood.eval()
# model.eval()

# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     # manifold
#     preds = likelihood(model(noisy_x))
#     posterior_mean = preds.mean.mul(std_l).add(mu_l).cpu().numpy()
#     posterior_std = preds.stddev.mul(std_l).add(mu_l).cpu().numpy()
#     posterior_sample = preds.sample().mul(std_l).add(mu_l).cpu().numpy()
#     prior_var = kernel.base_kernel.variance(noisy_x).cpu().numpy()
#     kernel_eval = kernel(noisy_x[0, :].unsqueeze(0), noisy_x).evaluate().squeeze().cpu().numpy()
#     eigfunction = kernel.base_kernel.eigenvectors[:, 1].cpu().numpy()

#     # ambient
#     resolution = 100
#     y, x = torch.meshgrid(torch.linspace(-1.5, 1.5, resolution), torch.linspace(-1.5, 1.5, resolution))
#     grid_x = torch.stack((torch.ravel(x), torch.ravel(y)), dim=1).to(device).requires_grad_(True)
#     preds = likelihood(model(grid_x))
#     posterior_mean_grid = preds.mean.mul(std_l).add(mu_l).cpu().numpy()
#     posterior_std_grid = preds.stddev.mul(std_l).add(mu_l).cpu().numpy()
#     posterior_sample_grid = preds.sample().mul(std_l).add(mu_l).cpu().numpy()
#     prior_var_grid = kernel.base_kernel.variance(grid_x).cpu().numpy()
#     kernel_eval_grid = kernel(noisy_x[0, :].unsqueeze(0), grid_x).evaluate().squeeze().cpu().numpy()
#     eigfunctions_grid = kernel.base_kernel.eigenfunctions(grid_x)[1, :].cpu().numpy()

#     # Bring data to numpy
#     sampled_x = sampled_x.numpy()
#     sampled_y = sampled_y.numpy()
#     # noisy_x = noisy_x.cpu().mul(std_f).add(mu_f).numpy()
#     noisy_x = noisy_x.cpu().numpy()

# fig = plt.figure(figsize=(17, 9))

# # Ground Truth
# ax = fig.add_subplot(231)
# ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c=sampled_y, vmin=-0.5, vmax=0.5)
# ax.axis('equal')
# ax.set_title('Ground Truth')

# # Posterior Mean
# ax = fig.add_subplot(232)
# plot = ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c=posterior_mean, vmin=-0.5, vmax=0.5)
# fig.colorbar(plot)
# ax.axis('equal')
# ax.set_title('Posterior Mean')

# # Kernel evaluation
# ax = fig.add_subplot(233)
# plot = ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c=kernel_eval)
# ax.scatter(noisy_x[0, 0], noisy_x[0, 1], c='k', s=0.5)
# fig.colorbar(plot)
# ax.axis('equal')
# ax.set_title('Kernel Evaluation')

# # Posterior Sample
# ax = fig.add_subplot(234)
# plot = ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c=posterior_sample)
# fig.colorbar(plot)
# ax.axis('equal')
# ax.set_title('Posterior Sample')

# # Standard Deviation
# ax = fig.add_subplot(235)
# plot = ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c=posterior_std)
# fig.colorbar(plot)
# ax.axis('equal')
# ax.set_title('Standard Deviation')

# # Prior variance
# ax = fig.add_subplot(236)
# plot = ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c=prior_var)
# fig.colorbar(plot)
# ax.axis('equal')
# ax.set_title('Posterior Sample')


# fig = plt.figure(figsize=(17, 9))

# # Ground Truth
# ax = fig.add_subplot(231)
# ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c=sampled_y, vmin=-0.5, vmax=0.5)
# ax.axis('equal')
# ax.set_xlim(-1.5, 1.5)
# ax.set_ylim(-1.5, 1.5)
# ax.set_title('Ground Truth')

# # Posterior Mean
# ax = fig.add_subplot(232)
# contour = ax.contourf(x.cpu(), y.cpu(), posterior_mean_grid.reshape(resolution, -1), 500)
# # ax.contour(x.cpu(), y.cpu(), posterior_mean_grid.reshape(resolution, -1), 10, cmap=None, colors='#f2e68f')
# ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c='k', s=0.5)
# fig.colorbar(contour)
# ax.axis('square')
# ax.set_title('Posterior Mean')

# # Kernel evaluation
# ax = fig.add_subplot(233)
# contour = ax.contourf(x.cpu(), y.cpu(), kernel_eval_grid.reshape(resolution, -1), 500)
# # ax.contour(x.cpu(), y.cpu(), kernel_eval_grid.reshape(resolution, -1), 10, cmap=None, colors='#f2e68f')
# ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c='k', s=0.5)
# ax.scatter(noisy_x[0, 0], noisy_x[0, 1], c='k', s=1.0)
# fig.colorbar(contour)
# ax.axis('square')
# ax.set_title('Kernel Evaluation')

# # Posterior Sample
# ax = fig.add_subplot(234)
# contour = ax.contourf(x.cpu(), y.cpu(), posterior_sample_grid.reshape(resolution, -1), 500)
# # ax.contour(x.cpu(), y.cpu(), posterior_sample_grid.reshape(resolution, -1), 10, cmap=None, colors='#f2e68f')
# ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c='k', s=0.5)
# fig.colorbar(contour)
# ax.axis('square')
# ax.set_title('Posterior Sample')

# # Standard Deviation
# ax = fig.add_subplot(235)
# contour = ax.contourf(x.cpu(), y.cpu(), posterior_std_grid.reshape(resolution, -1), 500)
# # ax.contour(x.cpu(), y.cpu(), posterior_std_grid.reshape(resolution, -1), 10, cmap=None, colors='#f2e68f')
# ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c='k', s=0.5)
# fig.colorbar(contour)
# ax.axis('square')
# ax.set_title('Standard Deviation')

# # Prior variance
# ax = fig.add_subplot(236)
# contour = ax.contourf(x.cpu(), y.cpu(), prior_var_grid.reshape(resolution, -1), 500)
# # ax.contour(x.cpu(), y.cpu(), prior_var_grid.reshape(resolution, -1), 10, cmap=None, colors='#f2e68f')
# ax.scatter(noisy_x[:, 0], noisy_x[:, 1], c='k', s=0.5)
# fig.colorbar(contour)
# ax.axis('square')
# ax.set_title('Posterior Sample')

# plt.show(block=False)
