# !/usr/bin/env python
# encoding: utf-8

import scipy.spatial as ss
from gpytorch.priors import NormalPrior, GammaPrior
from manifold_gp.priors.inverse_gamma_prior import InverseGammaPrior
from manifold_gp.utils.file_read import get_data
from manifold_gp.utils.mesh_helper import groundtruth_from_samples
from manifold_gp.models.riemann_gp import RiemannGP
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from importlib.resources import files
import matplotlib.pyplot as plt
import matplotlib
import torch
import gpytorch
import numpy as np


data_path = files('manifold_gp.data').joinpath('dumbbell.msh')
data = get_data(data_path, "Nodes", "Elements")

vertices = data['Nodes'][:, 1:-1]
edges = data['Elements'][:, -2:].astype(int) - 1
truth, geodesics = groundtruth_from_samples(vertices, edges)

sampled_x = torch.from_numpy(vertices).float()
sampled_y = torch.from_numpy(truth).float()
(m, n) = sampled_x.shape

num_train = 10
num_test = 100
normalize_features = False
normalize_labels = True

noise_sampled_x = 0.0
noisy_x = sampled_x + noise_sampled_x * torch.randn(m, n)

torch.manual_seed(1337)
rand_idx = torch.randperm(m)
train_idx = rand_idx[:num_train]
train_x, train_y = noisy_x[train_idx, :], sampled_y[train_idx]

noise_train_y = 0.01
train_y += noise_train_y * torch.randn(num_train)

test_idx = rand_idx[num_train:num_train+num_test]
test_x, test_y = sampled_x[test_idx, :], sampled_y[test_idx]

noise_test_y = 0.0
test_y += noise_test_y * torch.randn(num_test)

if normalize_features:
    mu_x, std_x = noisy_x.mean(dim=-2, keepdim=True), train_x.std(dim=-2, keepdim=True) + 1e-6
    noisy_x.sub_(mu_x).div_(std_x)
    train_x.sub_(mu_x).div_(std_x)
    test_x.sub_(mu_x).div_(std_x)

if normalize_labels:
    mu_y, std_y = train_y.mean(), train_y.std()
    train_y.sub_(mu_y).div_(std_y)

neighbors = 10
kd_tree = ss.KDTree(vertices)
v = np.sort(kd_tree.query(vertices, k=neighbors+1)[0][:, 1:].ravel())
percentile_99 = int(np.round(v.shape[0]*0.99))
gamma_rate = 100.0/np.std(v)
gamma_concentration = gamma_rate * v[percentile_99] + 1
igamma_concentration = 100.0/np.std(v)
igamma_rate = (igamma_concentration+1)*v[percentile_99]

noisy_x, sampled_y = noisy_x.contiguous(), sampled_y.contiguous()
train_x, train_y = train_x.contiguous(), train_y.contiguous()
test_x, test_y = test_x.contiguous(), test_y.contiguous()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
noisy_x = noisy_x.to(device)
train_x, train_y = train_x.to(device), train_y.to(device)
test_x, test_y = test_x.to(device), test_y.to(device)

gamma_concentration, gamma_rate = torch.tensor([gamma_concentration]).to(device), torch.tensor([gamma_rate]).to(device)
igamma_concentration, igamma_rate = torch.tensor([igamma_concentration]).to(device), torch.tensor([igamma_rate]).to(device)

if normalize_features:
    mu_x, std_x = mu_x.to(device), std_x.to(device)

likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
    noise_prior=None  # NormalPrior(torch.tensor([0.0]).to(device),  torch.tensor([1/9]).sqrt().to(device))
)

kernel = gpytorch.kernels.ScaleKernel(
    RiemannMaternKernel(
        nu=1,
        nodes=noisy_x,
        neighbors=10,
        operator="randomwalk",
        modes=100,
        ball_scale=1.0,
        support_kernel=gpytorch.kernels.RBFKernel(),
        epsilon_prior=GammaPrior(gamma_concentration, gamma_rate),
        lengthscale_prior=None,  # InverseGammaPrior(igamma_concentration, igamma_rate)
    ),
    outputscale_prior=None  # NormalPrior(torch.tensor([1.0]).to(device),  torch.tensor([1/9]).sqrt().to(device))
)

model = RiemannGP(train_x, train_y, likelihood, kernel, train_idx).to(device)

hypers = {
    'likelihood.noise_covar.noise': 0.034,
    'covar_module.base_kernel.epsilon': 0.052,
    'covar_module.base_kernel.lengthscale': 8.114,
    'covar_module.outputscale': 0.221,
    'covar_module.base_kernel.support_kernel.lengthscale': 1.0,
}
model.initialize(**hypers)

likelihood.eval()
model.eval()

log_mse = []
log_nll = []
range_s = np.arange(0.001, 1.0, 0.001)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # for i in range_s:
    # print(i)
    # model.covar_module.outputscale = i  # torch.tensor(i)
    preds_test = likelihood(model(test_x))
    mean_test = preds_test.mean.mul(std_y).add(mu_y)
    error = test_y - mean_test
    covar = preds_test.covariance_matrix.mul(std_y.square())
    log_nll.append((0.5 * (torch.dot(error, torch.linalg.solve(covar, error)) + torch.logdet(covar) + error.size(-1) * np.log(2 * np.pi))/num_test).cpu().numpy())
    log_mse.append((error.square().sum().sqrt()/num_test).cpu().numpy())
    model._clear_cache()

model.covar_module(kernel.base_kernel.nodes, kernel.base_kernel.nodes, diag=True).mean()
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(range_s, np.array(log_mse))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(range_s, np.array(log_nll))

# plt.show()
