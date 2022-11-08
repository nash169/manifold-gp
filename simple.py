#!/usr/bin/env python
# encoding: utf-8

from turtle import distance, shape
import torch
import matplotlib.pyplot as plt
import numpy as np
import faiss
import faiss.contrib.torch_utils
import networkx as nx
from src.kernels.riemann_matern import RiemannMatern
from src.laplacian import Laplacian
from src.gaussian_process import GaussianProcess
from src.knn_expansion import KnnExpansion
from src.kernels.riemann_exp import RiemannExp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs
from src.laplacian_knn import LaplacianKnn

from src.matern_precision import MaternPrecision, PrecisionOperator

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
X = data[:20, :2]
Y = data[:20, -1][:, np.newaxis]

use_cuda = False  # torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

X_sampled = torch.from_numpy(X).float().to(device).requires_grad_(True)
Y_sampled = torch.from_numpy(Y).float().to(device).requires_grad_(True)

index = faiss.IndexFlatL2(X_sampled.shape[1])
index.train(X_sampled)
index.add(X_sampled)
k = 2
distances, neighbors = index.search(X_sampled, k+1)
distances = distances[:, 1:]
neighbors = neighbors[:, 1:]

lp = Laplacian(X_sampled)
# lp.train(Y_sampled, 10000)
for n, p in lp.named_parameters():
    print('Parameter name:', n)
    print(p.data)

lp_knn = LaplacianKnn(neighbors, distances)
# for n, p in lp_knn.named_parameters():
#     print('Parameter name:', n)
#     print(p.data)

phi = PrecisionOperator(MaternPrecision(X_sampled, 2, 1))

# # Training/Test points
# num_train = 50
# idx_train = torch.randint(X_sampled.shape[0], (num_train,))
# X_train = X_sampled[idx_train, :]
# Y_train = Y_sampled[idx_train]
# num_test = 10
# idx_test = torch.randint(X_sampled.shape[0], (num_test,))
# X_test = X_sampled[idx_test, :]
# Y_test = Y_sampled[idx_test]

# Build Diffusion Maps Laplacian
i = np.concatenate((np.repeat(np.arange(neighbors.shape[0]), neighbors.shape[1])[
    np.newaxis, :], neighbors.reshape(1, -1)), axis=0)
v = (X_sampled[i[0, :], :] - X_sampled[i[1, :], :]
     ).pow(2).sum(dim=1).div(-lp.eps_).exp()
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
L = (torch.sparse_coo_tensor(index_diag, torch.ones(neighbors.shape[0]), (
    neighbors.shape[0], neighbors.shape[0])).to(device) - torch.sparse.mm(D, L))/(1/4*lp.eps_)

# L = L.to_dense() + 2*lp_knn.nu_/lp_knn.k_**2*torch.eye(neighbors.shape[0])

# result = torch.dot(Y_sampled.squeeze(), torch.mv(
#     torch.matrix_power(L, lp_knn.nu_), Y_sampled.squeeze()))

# Get eigenvectors
num_eigs = 10
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
f.k = 2
f.sigma = torch.sqrt(lp.eps_/2)

# # Create Riemann Kernel
# kernel = RiemannMatern(lp.k_)
# # kernel = RiemannExp(lp.k_)
# kernel.eigenvalues = T
# kernel.eigenfunctions = f

# # GPR Riemann
# gp_r = GaussianProcess().to(device)
# gp_r.samples = X_train.to(device)
# gp_r.target = Y_train.to(device)
# gp_r.kernel_ = kernel
# gp_r.signal = (lp.sigma_, True)
# gp_r.noise = (lp.sigma_n_.data, True)
# gp_r.update()
# # gp_r.train(10000)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(X[:, 0], X[:, 1])
# ax.scatter(X[0, 0], X[0, 1], c="r")
# ax.scatter(X[121, 0], X[121, 1], c="r")
# # X_train = X_train.cpu().detach().numpy()
# # ax.scatter(X_train[:, 0], X_train[:, 1], c="r", edgecolors="r")
# ax.axis('equal')

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plot = ax.scatter(X[:, 0], X[:, 1], c=Y, vmin=-0.5, vmax=0.5)
# fig.colorbar(plot)
# ax.axis('equal')
# ax.set_title('Ground Truth')


# fig = plt.figure()
# ax = fig.add_subplot(111)
# sol = gp_r(X_sampled).squeeze().cpu().detach().numpy()
# plot = ax.scatter(X[:, 0], X[:, 1], c=sol, vmin=-0.5, vmax=0.5)
# fig.colorbar(plot)
# ax.axis('equal')
# ax.set_title('Riemann GPR')

# plt.show()
