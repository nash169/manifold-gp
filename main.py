#!/usr/bin/env python
# encoding: utf-8

import faiss
import faiss.contrib.torch_utils
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.utils import build_ground_truth, plot_function, squared_exp, lanczos
from src.kernels.riemann_exp import RiemannExp
from src.kernels.squared_exp import SquaredExp
from src.gaussian_process import GaussianProcess
from src.knn_expansion import KnnExpansion

# CPU/GPU setting
use_cuda = torch.cuda.is_available()
device = torch.device("cpu")  # torch.device("cuda" if use_cuda else "cpu")

# Load mesh
mesh = 'rsc/torus.msh'
nodes, faces, truth = build_ground_truth(mesh)
x, y, z = (nodes[:, i] for i in range(3))
# plot_function(x, y, z, faces, truth)
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
index = faiss.IndexFlatL2(X_sampled.shape[1])
index.train(X_sampled)
index.add(X_sampled)

# Build Graph
k = 50
sigma = 1e-1
eps = 2 * sigma**2
distances, neighbors = index.search(X_sampled, k)
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
L = (torch.sparse_coo_tensor(index_diag, torch.ones(neighbors.shape[0]), (
    neighbors.shape[0], neighbors.shape[0])).to(device) - torch.sparse.mm(D, L))/eps*4

# Get eigenvectors
num_eigs = 100
T, V = torch.lobpcg(L, k=num_eigs, largest=False)

# Create Riemann Kernel
kernel = RiemannExp()
kernel.eigenvalues = T

f = []
for i in range(num_eigs):
    f.append(KnnExpansion())
    f[-1].alpha = V[:, i]
    f[-1].knn = index
    f[-1].k = k
    f[-1].sigma = sigma

kernel.eigenfunctions = f

# GPR Riemann
gp_r = GaussianProcess()
gp_r.samples = X_train
gp_r.target = Y_train.unsqueeze(1)
gp_r.kernel_ = kernel
gp_r.signal = (torch.tensor(1.), True)
gp_r.noise = (torch.tensor(1e-3), True)

# opt_r = torch.optim.Adam(gp_r.parameters(), lr=1e-4)
# for i in range(1000):
#     K = gp_r.covariance()
#     loss = 0.5*(torch.mm(gp_r.target.t(), torch.linalg.solve(K, gp_r.target)) +
#                 torch.log(torch.linalg.det(K)))
#     loss.backward(retain_graph=True)
#     if i % 100 == 0:
#         print(
#             f"Iteration: {i}, Loss: {loss.item():0.2f}")
#     opt_r.step()
#     opt_r.zero_grad()

gp_r.update()
sol_r = gp_r(X_test).squeeze()
error_r = (Y_test - sol_r).abs().cpu().detach().numpy()
plot_function(x, y, z, faces, gp_r(X_sampled).squeeze().cpu().detach().numpy())

# GPR Ambient
gp_a = GaussianProcess()
gp_a.samples = X_train
gp_a.target = Y_train.unsqueeze(1)
gp_a.kernel_ = SquaredExp(5.)
gp_a.signal = (torch.tensor(0.1), True)
gp_a.noise = (torch.tensor(1.), True)

# opt_a = torch.optim.Adam(gp_a.parameters(), lr=1e-4)
# for i in range(1000):
#     K = gp_a.covariance()
#     loss = 0.5*(torch.mm(gp_a.target.t(), torch.linalg.solve(K, gp_a.target)) +
#                 torch.log(torch.linalg.det(K)))
#     loss.backward(retain_graph=True)
#     if i % 100 == 0:
#         print(
#             f"Iteration: {i}, Loss: {loss.item():0.2f}")
#     opt_a.step()
#     opt_a.zero_grad()

gp_a.update()
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

# # T, V = torch.linalg.eig(L.to_dense())
# plot_function(x, y, z, faces, V[:, 1].real.cpu().detach().numpy())
# plot_function(x, y, z, faces, V[:, 10].real.cpu().detach().numpy())

# T, V = lanczos(L, torch.rand(
#     L.shape[0], 1, dtype=torch.float64).to(device), 100)
