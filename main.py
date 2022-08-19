#!/usr/bin/env python
# encoding: utf-8

import faiss
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.utils import build_ground_truth, plot_function, squared_exp, lanczos
from src.kernels.riemann_exp import RiemannExp
from src.kernels.squared_exp import SquaredExp
from src.gaussian_process import GaussianProcess

# CPU/GPU setting
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load mesh
mesh = 'rsc/torus.msh'
nodes, faces, truth = build_ground_truth(mesh)
# x, y, z = (nodes[:, i] for i in range(3))
# plot_function(x, y, z, faces, truth)
X = np.array(nodes, dtype=np.float32)
Y = np.array(truth, dtype=np.float32)
X_sampled = torch.from_numpy(X).float().to(device).requires_grad_(True)
Y_sampled = torch.from_numpy(Y).float().to(device).requires_grad_(True)

# Training/Test points
num_train = 100
idx_train = torch.randint(X_sampled.shape[0], (num_train,))
X_train = X_sampled[idx_train, :]
Y_train = Y_sampled[idx_train]
num_test = 100
idx_test = torch.randint(X_sampled.shape[0], (num_test,))
X_test = X_sampled[idx_test, :]
Y_test = Y_sampled[idx_test]

# Build Graph
k = 50
index = faiss.index_factory(X.shape[1], "Flat")
index.train(X)
index.add(X)
distances, neighbors = index.search(X, k)
i = np.concatenate((np.repeat(np.arange(neighbors.shape[0]), neighbors.shape[1])[
    np.newaxis, :], neighbors.reshape(1, -1)), axis=0)
v = np.ones(i.shape[1])
G = torch.sparse_coo_tensor(
    i, v, (neighbors.shape[0], neighbors.shape[0])).to(device)

# Build Diffusion Maps Laplacian (very inefficient way...)
sigma = 1e-2
eps = 2 * sigma**2
L = squared_exp(X_sampled, X_sampled, sigma)*G.to_dense()
D = L.sum(dim=1).pow(-1).diag()
L = torch.mm(D, torch.mm(L, D))
L = (torch.eye(L.shape[0]).to(device) -
     torch.mm(L.sum(dim=1).pow(-1).diag(), L)) / eps * 4

T, V = lanczos(L, torch.rand(
    L.shape[0], 1, dtype=torch.float64).to(device), 100)

# GPR Riemann
gp_r = GaussianProcess()
gp_r.samples = X_train
gp_r.target = Y_train.unsqueeze(1)
gp_r.kernel_ = RiemannExp()
gp_r.kernel_.samples = X_sampled
gp_r.kernel_.eigenvalues = T.diag()
gp_r.kernel_.eigenvectors = V
gp_r.signal = (torch.tensor(1.), True)
gp_r.noise = (torch.tensor(1e-3), True)

gp_r.update()
sol_r = gp_r(X_test).squeeze()
error_r = (Y_test - sol_r).abs().cpu().detach().numpy()

# GPR Ambient
gp_a = GaussianProcess()
gp_a.samples = X_train
gp_a.target = Y_train.unsqueeze(1)
gp_a.kernel_ = SquaredExp()
gp_a.signal = (torch.tensor(1.), True)
gp_a.noise = (torch.tensor(1e-3), True)

gp_a.update()
sol_a = gp_a(X_test).squeeze()
error_a = (Y_test - sol_a).abs().cpu().detach().numpy()

print("Ambient GPR -> mean: ", error_a.mean(), " std: ", error_a.std())
print("Riemann GPR -> mean: ", error_r.mean(), " std: ", error_r.std())

# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(num_test), error_r, '-o')
ax.plot(np.arange(num_test), error_a, '-o')

plt.show()
