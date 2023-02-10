# !/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import gpytorch

from importlib.resources import files
from manifold_gp.utils.generate_truth import groundtruth_from_mesh

from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.models.riemann_gp import RiemannGP

import faiss
import faiss.contrib.torch_utils

from torch_geometric.utils import coalesce, scatter
from torch_scatter import scatter_add

# Set device
use_cuda = False  # torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load mesh and generate ground truth
data_path = files('manifold_gp.data').joinpath('dragon10k.stl')
nodes, faces, truth = groundtruth_from_mesh(data_path)
nodes = torch.from_numpy(nodes).float().to(device)[:10, :]

# KNN Faiss
knn = faiss.IndexFlatL2(nodes.shape[1])

# Initialize kernel
neighbors = 5
modes = 5
nu = 1
kernel = gpytorch.kernels.ScaleKernel(RiemannMaternKernel(
    nu=nu, nodes=nodes, neighbors=neighbors, modes=modes))
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-8))
model = RiemannGP(nodes, truth, likelihood, kernel)

neighbors = 5
epsilon = kernel.base_kernel.epsilon  # torch.tensor([[0.05]])
lengthscale = kernel.base_kernel.lengthscale
outputscale = kernel.outputscale
knn.reset()
knn.train(nodes)
knn.add(nodes)
val, idx = knn.search(nodes, neighbors+1)
val = val[:, 1:]
idx = idx[:, 1:]


def vanilla(idx, val):
    dim = idx.shape[0]
    rows = torch.arange(idx.shape[0]).repeat_interleave(idx.shape[1])
    cols = idx.reshape(1, -1).squeeze()
    val = val.reshape(1, -1).squeeze()

    split = cols > rows
    rows, cols = torch.cat([rows[split], cols[~split]], dim=0), torch.cat(
        [cols[split], rows[~split]], dim=0)
    idx = torch.stack([rows, cols], dim=0)
    val = torch.cat([val[split], val[~split]])
    idx, val = coalesce(idx, val, reduce='mean')
    rows = idx[0, :]
    cols = idx[1, :]

    val = val.div(-2*epsilon.square()).exp().squeeze()

    Lsparse = torch.sparse_coo_tensor(idx, val, (dim, dim))
    L = (Lsparse.to_dense() + Lsparse.to_dense().t())
    D = L.sum(dim=1).pow(-1).diag()
    L = torch.mm(D, torch.mm(L, D))

    D = L.sum(dim=1).pow(-0.5).diag()
    L = torch.mm(D, torch.mm(L, D))
    # L = torch.mm(D, L)
    # L *= D.unsqueeze(-1)

    return torch.eye(dim)-L


def sparse(idx, val):
    dim = idx.shape[0]
    rows = torch.arange(idx.shape[0]).repeat_interleave(idx.shape[1])
    cols = idx.reshape(1, -1).squeeze()
    val = val.reshape(1, -1).squeeze()

    split = cols > rows
    rows, cols = torch.cat([rows[split], cols[~split]], dim=0), torch.cat(
        [cols[split], rows[~split]], dim=0)
    idx = torch.stack([rows, cols], dim=0)
    val = torch.cat([val[split], val[~split]])
    idx, val = coalesce(idx, val, reduce='mean')
    rows = idx[0, :]
    cols = idx[1, :]

    # row_idx, row_count = torch.unique(idx[0, :], return_counts=True)

    val = val.div(-2*epsilon.square()).exp().squeeze()
    # deg = scatter(val, idx[0, :])
    # + torch.zeros(dim).scatter_add_(0, cols, val)
    deg = torch.zeros(dim).scatter_add(0, rows, val).scatter_add(0, cols, val)
    # deg = torch.sum(torch.zeros(2, dim).scatter_add(
    #     1, idx, val.repeat(2, 1)), dim=0)
    # deg += torch.zeros(dim).scatter_add_(0, cols, val)

    # val = val*deg.repeat_interleave(row_count)
    # idx_col = idx[1, :][torch.isin(idx[1, :], row_idx)]
    # val[idx_col] *= deg[idx_col]

    val = val.div(deg[rows]*deg[cols])
    # val /= deg[rows]*deg[cols]
    deg = torch.zeros(dim).scatter_add(
        0, rows, val).scatter_add(0, cols, val).sqrt()
    val.div_(deg[rows]*deg[cols])

    # val /= deg[cols]

    # L = torch.sparse_coo_tensor(idx, val, (dim, dim))
    # L = L.to_dense()  # + L.to_dense().t()

    return val, deg, rows, cols


def print_mat(mat):
    s = [[str(e) for e in row] for row in mat]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


M = vanilla(idx, val) + \
    torch.eye(nodes.shape[0])*2*nu/lengthscale.square().squeeze()
v, d, r, c = sparse(idx, val)
# v = sparse(idx, val)

y = torch.rand(nodes.shape[0])
mv = torch.mv(M, y)
# mv2 = y - (torch.zeros(10).scatter_add_(
#     0, r, v*y[c]) + torch.zeros(10).scatter_add_(0, c, v*y[r]))

# mv2 = y.scatter_add(0, r, -v*y[c]).scatter_add(0, c, -v*y[r])

# mv2 = torch.mv(L.t(), y)
z = torch.rand(nodes.shape[0], 3)
mv3 = torch.zeros(z.shape).scatter_add_(
    0, r.unsqueeze(-1).repeat(1, 3), z[c]*v.unsqueeze(-1))

# opt = kernel.precision()
