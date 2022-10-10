#!/usr/bin/env python
# encoding: utf-8

import torch
import matplotlib.pyplot as plt
import numpy as np
import faiss
import faiss.contrib.torch_utils
import networkx as nx
from src.laplacian import Laplacian
from src.gaussian_process import GaussianProcess
from src.knn_expansion import KnnExpansion
from src.kernels.riemann_exp import RiemannExp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs


# def ground_truth(X):
#     import faiss
#     index = faiss.IndexFlatL2(X.shape[1])
#     index.train(X)
#     index.add(X)
#     distances, neighbors = index.search(X, 3)

#     graph = nx.Graph()
#     for i in range(X.shape[0]):
#         graph.add_node(i, pos=(X[i, 0], X[i, 1]))
#     for i in range(X.shape[0]):
#         graph.add_edge(i, neighbors[i, 1], length=distances[i, 1])
#         graph.add_edge(i, neighbors[i, 2], length=distances[i, 2])

#     geodesics = nx.shortest_path_length(graph, source=0, weight='length')
#     Y = np.zeros((X.shape[0]))
#     for i in range(X.shape[0]):
#         Y[i] = 0.5*np.sin(5e2 * geodesics.get(i)**2)

#     return Y


# X = np.loadtxt('rsc/dumbbell.msh')[:, 1:]
# Y = ground_truth(X)[:, np.newaxis]
# np.savetxt('rsc/dumbbell.msh', np.concatenate((X, Y), axis=1))

data = np.loadtxt('rsc/dumbbell.msh')
X = data[:, :3]
Y = data[:, -1][:, np.newaxis]

use_cuda = False  # torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

X_sampled = torch.from_numpy(X[:, :2]).float().to(device).requires_grad_(True)
Y_sampled = torch.from_numpy(Y).float().to(device).requires_grad_(True)

lp = Laplacian()
lp.samples = X_sampled
lp.train(Y_sampled)

# Training/Test points
num_train = 50
idx_train = torch.randint(X_sampled.shape[0], (num_train,))
X_train = X_sampled[idx_train, :]
Y_train = Y_sampled[idx_train]
num_test = 10
idx_test = torch.randint(X_sampled.shape[0], (num_test,))
X_test = X_sampled[idx_test, :]
Y_test = Y_sampled[idx_test]

# Build Diffusion Maps Laplacian
index = faiss.IndexFlatL2(X_sampled.shape[1])
index.train(X_sampled)
index.add(X_sampled)
k = 50
eps = 2*lp.eps_**2
distances, neighbors = index.search(X_sampled, k+1)
distances = distances[:, 1:]
neighbors = neighbors[:, 1:]
i = np.concatenate((np.repeat(np.arange(neighbors.shape[0]), neighbors.shape[1])[
    np.newaxis, :], neighbors.reshape(1, -1)), axis=0)
v = (X_sampled[i[0, :], :] - X_sampled[i[1, :], :]
     ).pow(2).sum(dim=1).div(-eps).exp()
L = torch.sparse_coo_tensor(
    i, v, (neighbors.shape[0], neighbors.shape[0])).to(device)
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
f.sigma = lp.eps_

# Create Riemann Kernel
kernel = RiemannExp(lp.sigma_)
kernel.eigenvalues = T
kernel.eigenfunctions = f

# GPR Riemann
gp_r = GaussianProcess().to(device)
gp_r.samples = X_train.to(device)
gp_r.target = Y_train.to(device)
gp_r.kernel_ = kernel
gp_r.signal = (torch.tensor(1.), True)
gp_r.noise = (torch.tensor(1e-3), True)
# gp_r.update()
gp_r.train()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[0, 0], X[0, 1], X[0, 2], c="r", linewidths=10, edgecolors="r")
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
ax.scatter(X[:, 0], X[:, 1], Y, c=Y)
ax.set_box_aspect((np.ptp(X[:, 0]), np.ptp(X[:, 1]), np.ptp(Y)))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[0, 0], X[0, 1], X[0, 2], c="r", linewidths=10, edgecolors="r")
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
sol = gp_r(X_sampled).squeeze().cpu().detach().numpy()
ax.scatter(X[:, 0], X[:, 1], sol, c=sol)

plt.show()
