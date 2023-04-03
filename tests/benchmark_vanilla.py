# !/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import gpytorch
import random
import matplotlib.pyplot as plt

from importlib.resources import files

from manifold_gp.utils.torch_helper import memory_allocation

from scipy.io import loadmat
import scipy.spatial as ss

# get data
data = np.array(loadmat('manifold_gp/data/protein.mat')['data'])
cut = data.shape[0]  # 10000
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

# cut between 0.01 and 0.99 quantile of distances
kd_tree = ss.KDTree(train_x)
v = kd_tree.query(train_x, k=2)[0][:, 1]
idx = np.argsort(v)
percentile_1 = int(np.round(idx.shape[0]*0.00))
percentile_99 = int(np.round(idx.shape[0]*0.95))
train_x = train_x[idx[percentile_1:percentile_99], :]
train_y = train_y[idx[percentile_1:percentile_99]]

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

# make continguous
train_x, train_y = train_x.contiguous(), train_y.contiguous()
test_x, test_y = test_x.contiguous(), test_y.contiguous()

# bring to device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
train_x, train_y = train_x.to(device), train_y.to(device)
test_x, test_y = test_x.to(device), test_y.to(device)


# Model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.constant.requires_grad_(False)
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Training parameters
lr = 1e-2
iters = 1000
verbose = True
load = False
train = True

# Loop
count = 1
samples = 1
stats = torch.zeros(samples, 3).to(device)

for i in range(samples):
    print(f"Iteration: {count}/{samples}")

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-8)).cuda()
    model = ExactGPModel(train_x, train_y, likelihood).cuda()

    if load:
        model.load_state_dict(torch.load('outputs/models/vanilla.pth'))
    else:
        # Model Hyperparameters
        hypers = {
            'likelihood.noise_covar.noise': 1e-4,
            'covar_module.base_kernel.lengthscale': 1.0,
            'covar_module.outputscale': 1.0,
        }
        model.initialize(**hypers)

    if train:
        # Train model
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for k in range(iters):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            if verbose:
                print('Iter %d/%d - Loss: %.5f   lengthscale: %.5f   noise: %.5f' % (
                    k + 1, iters, loss.item(),
                    model.covar_module.base_kernel.lengthscale.item(),
                    model.likelihood.noise.item()))
            optimizer.step()

        torch.save(model.state_dict(), 'outputs/models/vanilla.pth')

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
# np.savetxt('outputs/riemann_'+str(nu)+'_'+str(neighbors) + '.csv', stats.detach().numpy())
