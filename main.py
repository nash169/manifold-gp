#!/usr/bin/env python
# encoding: utf-8

from yaml import load
import faiss
import faiss.contrib.torch_utils
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.utils import build_ground_truth, plot_function, squared_exp, lanczos, load_mesh
from src.kernels.riemann_exp import RiemannExp
from src.kernels.squared_exp import SquaredExp
from src.gaussian_process import GaussianProcess
from src.knn_expansion import KnnExpansion

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs

from time import time

# CPU/GPU setting
use_cuda = False  # torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# myMesh = myMesh.simplify_quadratic_decimation(2000)

# Load mesh
mesh = 'rsc/torus3k.msh'
nodes, faces, truth = build_ground_truth(mesh)
x, y, z = (nodes[:, i] for i in range(3))
plot_function(x, y, z, faces, truth)
X = np.array(nodes, dtype=np.float32)
Y = np.array(truth, dtype=np.float32)
X_sampled = torch.from_numpy(X).float().to(device).requires_grad_(True)
Y_sampled = torch.from_numpy(Y).float().to(device).requires_grad_(True)

# Training/Test points
num_train = 500
idx_train = torch.randint(X_sampled.shape[0], (num_train,))
X_train = X_sampled[idx_train, :]
Y_train = Y_sampled[idx_train]
num_test = 100
idx_test = torch.randint(X_sampled.shape[0], (num_test,))
X_test = X_sampled[idx_test, :]
Y_test = Y_sampled[idx_test]

# Set Faiss
if use_cuda:
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(X_sampled.shape[1])
    index = faiss.index_cpu_to_gpu(
        res, 0, index_flat)
else:
    index = faiss.IndexFlatL2(X_sampled.shape[1])
index.train(X_sampled)
index.add(X_sampled)

# Build Graph
k = 50
sigma = 5e-1  # opt 0.6757
eps = 2 * sigma**2
distances, neighbors = index.search(X_sampled, k+1)
distances = distances[:, 1:]
neighbors = neighbors[:, 1:]
i = np.concatenate((np.repeat(np.arange(neighbors.shape[0]), neighbors.shape[1])[
    np.newaxis, :], neighbors.reshape(1, -1)), axis=0)
v = (X_sampled[i[0, :], :] - X_sampled[i[1, :], :]
     ).pow(2).sum(dim=1).div(-eps).exp()
L = torch.sparse_coo_tensor(
    i, v, (neighbors.shape[0], neighbors.shape[0])).to(device)

# Build Diffusion Maps Laplacian
D = torch.sparse.sum(L, dim=1).pow(-1)
index_diag = torch.cat((D.indices(), D.indices()), dim=0)
D = torch.sparse_coo_tensor(index_diag, D.values(
), (neighbors.shape[0], neighbors.shape[0])).to(device)
L = torch.sparse.mm(D, torch.sparse.mm(L, D))

D = torch.sparse.sum(L, dim=1).pow(-1)
D = torch.sparse_coo_tensor(index_diag, D.values(
), (neighbors.shape[0], neighbors.shape[0])).to(device)
L = torch.sparse_coo_tensor(index_diag, torch.ones(neighbors.shape[0]), (
    neighbors.shape[0], neighbors.shape[0])).to(device) - torch.sparse.mm(D, L)

# Get eigenvectors
num_eigs = 100
# T, V = torch.lobpcg(L, k=num_eigs, largest=False)
# T, V = torch.linalg.eig(L.to_dense())
# T, V = lanczos(L, torch.rand(
#     L.shape[0], 1, dtype=torch.float64).to(device), 100)

indices = L.coalesce().indices().cpu().detach().numpy()
values = L.coalesce().values().cpu().detach().numpy()
Ls = coo_matrix((values, (indices[0, :], indices[1, :])), shape=L.shape)
T, V = eigs(Ls, k=num_eigs, which='SR')

T = torch.from_numpy(T).float().to(device).requires_grad_(True)
V = torch.from_numpy(V).float().to(device).requires_grad_(True)

# Create KNN Eigenfunctions
f = KnnExpansion()
f.alpha = V
f.knn = index
f.k = k
f.sigma = sigma

# Create Riemann Kernel
kernel = RiemannExp(1e-2)
kernel.eigenvalues = T
kernel.eigenfunctions = f

# GPR Riemann
gp_r = GaussianProcess().to(device)
gp_r.samples = X_train.to(device)
gp_r.target = Y_train.unsqueeze(1).to(device)
gp_r.kernel_ = kernel
gp_r.signal = (torch.tensor(1.), True)
gp_r.noise = (torch.tensor(1e-3), True)
gp_r.train()
sol_r = gp_r(X_test).squeeze()
error_r = (Y_test - sol_r).abs().cpu().detach().numpy()
plot_function(x, y, z, faces, gp_r(X_sampled).squeeze().cpu().detach().numpy())

# GPR Ambient
gp_a = GaussianProcess()
gp_a.samples = X_train
gp_a.target = Y_train.unsqueeze(1)
gp_a.kernel_ = SquaredExp(1.)
gp_a.signal = (torch.tensor(0.1), True)
gp_a.noise = (torch.tensor(1.), True)
gp_r.train()
sol_a = gp_a(X_test).squeeze()
error_a = (Y_test - sol_a).abs().cpu().detach().numpy()
plot_function(x, y, z, faces, gp_a(X_sampled).squeeze().cpu().detach().numpy())

print("Ambient GPR -> mean: ", error_a.mean(), " std: ", error_a.std())
print("Riemann GPR -> mean: ", error_r.mean(), " std: ", error_r.std())

# Plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(num_test), error_r, '-o')
ax.plot(np.arange(num_test), error_a, '-o')

plt.show()
