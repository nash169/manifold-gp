# !/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import gpytorch
import random
import matplotlib.pyplot as plt
import faiss

from importlib.resources import files

from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.models.riemann_gp import RiemannGP

from manifold_gp.priors.inverse_gamma_prior import InverseGammaPrior
from gpytorch.priors import NormalPrior, GammaPrior

# from manifold_gp.utils.generate_truth import groundtruth_from_mesh
# from manifold_gp.utils.torch_helper import memory_allocation
# from torch.distributions import Normal, Gamma
# from manifold_gp.utils.atom3d_dataset import Atom3D, TransformSMP

import os
from scipy.io import loadmat
import scipy.spatial as ss
import urllib.request

# datasets
datasets = {
    'protein': 'https://drive.google.com/uc?export=download&id=1nRb8e7qooozXkNghC5eQS0JeywSXGX2S',
    'elevators': 'https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk'
}

# get data
dataset = 'protein'
if not os.path.isfile(f'manifold_gp/data/{dataset}.mat'):
    print(f'Downloading \'{dataset}\' UCI dataset...')
    urllib.request.urlretrieve(datasets[dataset], f'manifold_gp/data/{dataset}.mat')
data = np.array(loadmat('manifold_gp/data/protein.mat')['data'])
cut = 10000
sampled_x, sampled_y = data[:cut, :-1], data[:cut, -1]

# remove coincident points
sampled_x, id_unique = np.unique(sampled_x, axis=0, return_index=True)
sampled_y = sampled_y[id_unique]
m = sampled_x.shape[0]

# make train/val/test
n_train = int(0.8 * m)
idx_rand = np.arange(m)  # np.random.permutation(m)
train_x, train_y = sampled_x[idx_rand[:n_train], :], sampled_y[idx_rand[:n_train]]
test_x, test_y = sampled_x[idx_rand[n_train:], :], sampled_y[idx_rand[n_train:]]

# cut between 0.01 and 0.99 quantile of distances and calculate concentration and rate parameters for gamma/igamma
kd_tree = ss.KDTree(train_x)
v = kd_tree.query(train_x, k=2)[0][:, 1]
idx = np.argsort(v)
percentile_1 = int(np.round(idx.shape[0]*0.00))
percentile_95 = int(np.round(idx.shape[0]*0.95))
train_x = train_x[idx[percentile_1:percentile_95], :]
train_y = train_y[idx[percentile_1:percentile_95]]
del percentile_95, percentile_1, idx, v, kd_tree

# convert to torch
train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()

# normalize features
mean = train_x.mean(dim=-2, keepdim=True)
std = train_x.std(dim=-2, keepdim=True) + 1e-6  # prevent dividing by 0
train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

# normalize labels
mean, std = train_y.mean(), train_y.std()
train_y = (train_y - mean) / std
test_y = (test_y - mean) / std

# make contiguous
train_x, train_y = train_x.contiguous(), train_y.contiguous()
test_x, test_y = test_x.contiguous(), test_y.contiguous()

# kernel parameters
nu = 3
neighbors = 50
modes = 100

# priors parameters
kd_tree = ss.KDTree(train_x.numpy())
v = np.sort(kd_tree.query(train_x, k=neighbors+1)[0][:, 1:].ravel())
percentile_99 = int(np.round(v.shape[0]*0.99))
gamma_rate = 100.0/np.std(v)
gamma_concentration = gamma_rate * v[percentile_99] + 1
igamma_concentration = 100.0/np.std(v)
igamma_rate = (igamma_concentration+1)*v[percentile_99]
del percentile_99, v, kd_tree

# bring to device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
train_x, train_y = train_x.to(device), train_y.to(device)
test_x, test_y = test_x.to(device), test_y.to(device)
gamma_concentration, gamma_rate = torch.tensor([gamma_concentration]).to(device), torch.tensor([gamma_rate]).to(device)
igamma_concentration, igamma_rate = torch.tensor([igamma_concentration]).to(device), torch.tensor([igamma_rate]).to(device)

# kernel
kernel = gpytorch.kernels.ScaleKernel(
    RiemannMaternKernel(
        nu=nu,
        nodes=train_x,
        neighbors=neighbors,
        modes=modes,
        support_kernel=gpytorch.kernels.RBFKernel(),
        epsilon_prior=GammaPrior(gamma_concentration, gamma_rate),
        lengthscale_prior=InverseGammaPrior(igamma_concentration, igamma_rate)
    ),
    outputscale_prior=NormalPrior(torch.tensor([1.0]).to(device),  torch.tensor([1/9]).sqrt().to(device))
)

# likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
    noise_prior=NormalPrior(torch.tensor([0.0]).to(device),  torch.tensor([1/9]).sqrt().to(device)))

# # Distance analysis
# fig = plt.figure()
# ax = fig.add_subplot(111)
# val = kernel.base_kernel.values.sqrt()
# # val, idx = kernel.base_kernel.knn.search(train_x, neighbors+1)
# # val = val[:, 1].sqrt()
# x = torch.linspace(0.001, val.max().cpu(), 1000).to(device)
# values_sorted = torch.sort(val)[0]
# quantile_95 = int(np.round(values_sorted.shape[0]*0.99))
# HIST_BINS = np.linspace(0, val.max().cpu(), 100)
# ax.hist(val.cpu(), HIST_BINS)
# ax.vlines(x=values_sorted[quantile_95].cpu(), ymin=0.0, ymax=val.shape[0]/10, ls='--', lw=2, alpha=0.5, colors='k')
# ax.vlines(x=kernel.base_kernel.eps_gte.cpu(), ymin=0.0, ymax=val.shape[0]/10, ls='--', lw=2, alpha=0.5, colors='r')
# ax.plot(x.cpu(), 10000*kernel.base_kernel.epsilon_prior.log_prob(x).exp().cpu(), 'k')
# ax.plot(x.cpu(), 10000*GammaPrior(gamma_concentration, gamma_rate).log_prob(x).exp().cpu(), 'k')
# plt.show(block=False)

# training parameters
lr = 1e-1
iters = 1
verbose = True
load = False
train = True

# evaluation parameters
count = 1
samples = 1
stats = torch.zeros(samples, 3).to(device)

for i in range(samples):
    print(f"Iteration: {count}/{samples}")

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-8))
    model = RiemannGP(train_x, train_y, likelihood, kernel).to(device)

    if load:
        model.load_state_dict(torch.load('outputs/models/riemann_'+str(nu)+'_'+str(neighbors) + '.pth', map_location=torch.device(train_x.device)))
    else:
        # Model Hyperparameters
        # torch.manual_seed(1337)
        hypers = {
            'likelihood.noise_covar.noise': 1e-4,
            'covar_module.base_kernel.epsilon': 1.0,
            'covar_module.base_kernel.lengthscale': 1.0,
            'covar_module.outputscale': 1.0,
            'covar_module.base_kernel.support_kernel.lengthscale': 0.5
        }
        model.initialize(**hypers)

    if train:
        # Train model
        model.manifold_informed_train(lr, iters, verbose)

        # Save model
        torch.save(model.state_dict(), 'outputs/models/riemann_'+str(nu)+'_'+str(neighbors) + '.pth')

    # Model Evaluation
    likelihood.eval()
    model.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(test_x))
        # mean = preds.mean*std_n + mu_n
        # std = (preds.stddev*std_n + mu_n).abs()
        mean = preds.mean
        std = preds.stddev.abs()
        error = test_y - mean

        stats[i, 0] = error.abs().mean()
        stats[i, 1] = std.abs().mean()
        stats[i, 2] = preds.log_prob(test_y)

    count += 1

cal = torch.sum(error.abs() <= 1.96*std)/test_y.shape[0] * 100
np.savetxt('outputs/riemann_'+str(nu)+'_'+str(neighbors) + '.csv', stats.detach().cpu().numpy())
