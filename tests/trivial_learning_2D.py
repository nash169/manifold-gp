# !/usr/bin/env python
# encoding: utf-8

from time import time
from manifold_gp.utils.sparse_operator import SparseOperator
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

import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs

# Set device
use_cuda = False  # torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load mesh and generate ground truth
data_path = files('manifold_gp.data').joinpath('dragon10k.stl')
nodes, faces, truth = groundtruth_from_mesh(data_path)
sampled_x = torch.from_numpy(nodes).float().to(device)[:10, :]
# sampled_x = (sampled_x - torch.mean(sampled_x, dim=0)) / \
#     torch.std(sampled_x, dim=0)
sampled_y = torch.from_numpy(truth).float().to(device)[:10]

# sampled_x = sampled_x[:20, :]
# sampled_y = sampled_y[:20]
(m, n) = sampled_x.shape

# Noisy dataset
manifold_noise = 0.0
noisy_x = sampled_x + manifold_noise * torch.randn(m, n).to(device)
function_noise = 0.0
noisy_y = sampled_y + function_noise * torch.randn(m).to(device)

# Initialize kernel
nu = 3
neighbors = 5
modes = 5
alpha = 1
laplacian = "normalized"
kernel = gpytorch.kernels.ScaleKernel(RiemannMaternKernel(
    nu=nu, nodes=noisy_x, neighbors=neighbors, modes=modes, alpha=1, laplacian=laplacian))

# Initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-8))
model = RiemannGP(noisy_x, noisy_y, likelihood, kernel).to(device)

# Model Hyperparameters
hypers = {
    'likelihood.noise_covar.noise': 0.005**2,  # 1e-5,
    'covar_module.base_kernel.epsilon': 0.09,  # 0.5,
    'covar_module.base_kernel.lengthscale': 5,  # 1.603,  # 0.5,
    'covar_module.outputscale': 1.552**2,  # 1.0,
}
model.initialize(**hypers)

# Train model
lr = 1e-1
iters = 500
verbose = True
# loss = model.manifold_informed_train(lr, iters, verbose)

# Model Evaluation
likelihood.eval()
model.eval()

# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     preds = likelihood(model(noisy_x))
#     mean = preds.mean
#     std = preds.stddev
#     posterior_sample = preds.sample()
#     point = 3000
#     kernel_eval = kernel(sampled_x[point, :].unsqueeze(
#         0), sampled_x).evaluate().squeeze()
#     eigfunctions = kernel.base_kernel.eigenfunctions(sampled_x)

#     # Bring data to numpy
#     sampled_x = sampled_x.cpu().numpy()
#     sampled_y = sampled_y.cpu().numpy()
#     kernel_eval = kernel_eval.cpu().numpy()
#     eigfunctions = eigfunctions.cpu().numpy()
#     posterior_sample = posterior_sample.cpu().numpy()
#     mean = mean.cpu().numpy()
#     std = std.cpu().numpy()

# data_path = files('manifold_gp.data').joinpath('dragon100k.msh')
# nodes, faces2, _ = groundtruth_from_mesh(data_path)
# test_x = torch.from_numpy(nodes).float().to(device)

# with torch.no_grad():
#     eigfunctions = kernel.base_kernel.eigenfunctions(test_x).cpu().numpy()
#     point = 3405  # 3366
#     kernel_eval = kernel(sampled_x[point, :].unsqueeze(
#         0), sampled_x).evaluate().squeeze().cpu().numpy()
#     point2 = 67757
#     kernel_eval2 = kernel(test_x[point2, :].unsqueeze(
#         0), test_x).evaluate().squeeze().cpu().numpy()
#     sampled_x = sampled_x.cpu().numpy()
#     test_x = test_x.cpu().numpy()

# v_options = {'mode': 'sphere', 'scale_factor': 3e-3, 'color': (0, 0, 0)}

# # Ground Truth
# mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1],
#                      sampled_x[:, 2], faces, scalars=sampled_y)
# mlab.colorbar(orientation='vertical')

# # Mean
# mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1],
#                      sampled_x[:, 2], faces, scalars=mean)
# mlab.colorbar(orientation='vertical')

# # Standard Deviation
# mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1],
#                      sampled_x[:, 2], faces, scalars=std)
# mlab.colorbar(orientation='vertical')

# # One Posterior Sample
# mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1],
#                      sampled_x[:, 2], faces, scalars=posterior_sample)
# mlab.colorbar(orientation='vertical')

# # Kernel evaluation
# kernel_eval = kernel_eval - np.min(kernel_eval)
# kernel_eval /= np.max(kernel_eval)
# mlab.figure(size=(1920, 1360), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1],
#                      sampled_x[:, 2], faces, scalars=kernel_eval)
# mlab.colorbar(orientation='vertical')
# mlab.points3d(sampled_x[point, 0], sampled_x[point, 1],
#               sampled_x[point, 2], **v_options)
# mlab.view(0.0, 180.0, 0.5139171204775793)

# kernel_eval2 = kernel_eval2 - np.min(kernel_eval2)
# kernel_eval2 /= np.max(kernel_eval2)
# mlab.figure(size=(1920, 1360), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# mlab.triangular_mesh(test_x[:, 0], test_x[:, 1],
#                      test_x[:, 2], faces2, scalars=kernel_eval2)
# mlab.colorbar(orientation='vertical')
# mlab.points3d(test_x[point2, 0], test_x[point2, 1],
#               test_x[point2, 2], **v_options)
# mlab.view(0.0, 180.0, 0.5139171204775793)

# # Eigenfunctions
# mode = 50
# eigfun = eigfunctions[mode, :] - np.min(eigfunctions[mode, :])
# eigfun /= np.max(eigfun)
# mlab.figure(size=(1920, 1360), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# mlab.triangular_mesh(test_x[:, 0], test_x[:, 1],
#                      test_x[:, 2], faces2, scalars=eigfun)
# mlab.colorbar(orientation='vertical')
# mlab.view(0.0, 180.0, 0.5139171204775793)
# # mlab.savefig('dragon_eigfun1_100k.png')

# eigvec = kernel.base_kernel.eigenvectors[:, mode].cpu().numpy(
# ) - np.min(kernel.base_kernel.eigenvectors[:, mode].cpu().numpy())
# eigvec /= np.max(eigvec)
# mlab.figure(size=(1920, 1360), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# mlab.triangular_mesh(sampled_x[:, 0], sampled_x[:, 1], sampled_x[:, 2],
#                      faces, scalars=eigvec)
# mlab.colorbar(orientation='vertical')
# mlab.view(0.0, 180.0, 0.5139171204775793)
# # mlab.savefig('dragon_eigfun1_5k.png')


# val = kernel.base_kernel.laplacian(alpha=1, type="normalized").double()
# idx = kernel.base_kernel.indices

# L_opt = SparseOperator(val, idx, torch.Size([val.shape[0], val.shape[0]]))
# L_sparse = kernel.base_kernel.to_sparse(val)
# L_dense = L_sparse.to_dense()
# # L_dense = (L_dense + L_dense.t())/2
# # L_dense = torch.triu(L_dense, diagonal=1) + torch.triu(L_dense,
# #                                                        diagonal=1).t() + torch.eye(val.shape[0], val.shape[0]).to(device)

# indices = L_sparse.coalesce().indices().cpu().detach().numpy()
# values = L_sparse.coalesce().values().cpu().detach().numpy()
# L_scipy = coo_matrix(
#     (values, (indices[0, :], indices[1, :])), shape=L_sparse.shape)
# L_scipy_dense = L_scipy.todense()

# t0 = time()
# evals, evecs = torch.linalg.eig(L_dense)
# # evals, evecs = scipy.sparse.linalg.eigsh(L_scipy, k=100, which='SM')
# # evals, evecs = np.linalg.eig((L_scipy_dense + L_scipy_dense.T)/2)
# # with gpytorch.settings.max_root_decomposition_size(val.shape[0]):
# #     evals, evecs = L_opt.diagonalization()
# t1 = time()
# print("Torch eigh: %.4g sec" % (t1 - t0))

# # mlab.figure(size=(1920, 1360), fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
# # mlab.triangular_mesh(sampled_x.cpu().numpy()[:, 0], sampled_x.cpu().numpy()[:, 1], sampled_x.cpu().numpy()[:, 2],
# #                      faces, scalars=evecs[:, 1])
# # mlab.colorbar(orientation='vertical')
