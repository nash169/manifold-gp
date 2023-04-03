# !/usr/bin/env python
# encoding: utf-8

from time import time
import math

import numpy as np
import torch
import gpytorch

from importlib.resources import files
from manifold_gp.utils.generate_truth import groundtruth_from_mesh

from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.models.riemann_gp import RiemannGP, ScaleWrapper, NoiseWrapper, SemiSupervisedWrapper

import faiss
import faiss.contrib.torch_utils

from torch_geometric.utils import coalesce
from linear_operator import to_linear_operator

# Set device
use_cuda = False  # torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load mesh and generate ground truth
data_path = files('manifold_gp.data').joinpath('dragon10k.stl')
nodes, faces, truth = groundtruth_from_mesh(data_path)
nodes = torch.from_numpy(nodes).float().to(device)[:400, :]
truth = torch.from_numpy(truth).float().to(device)[:400]

# train
torch.manual_seed(1337)
y = truth  # torch.rand(nodes.shape[0])
semi_idx = torch.randperm(nodes.shape[0])
labeled = semi_idx[:100]
not_labeled = semi_idx[100:]
z = y[labeled]
x = nodes[labeled, :]

# KNN Faiss
knn = faiss.IndexFlatL2(nodes.shape[1])

# Initialize kernel
neighbors = 5
modes = 5
nu = 3
kernel = gpytorch.kernels.ScaleKernel(RiemannMaternKernel(
    nu=nu, nodes=nodes, neighbors=neighbors, modes=modes))
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-8))
# model = RiemannGP(x, z, likelihood, kernel, labeled)
model = RiemannGP(nodes, truth, likelihood, kernel)

kernel.base_kernel.epsilon = 0.5
epsilon = kernel.base_kernel.epsilon  # torch.tensor([[0.05]])
lengthscale = kernel.base_kernel.lengthscale
outputscale = kernel.outputscale
likelihood.noise = 1e-5
noise = likelihood.noise
knn.reset()
knn.train(nodes)
knn.add(nodes)
val, idx = knn.search(nodes, neighbors+1)
val = val[:, 1:]
idx = idx[:, 1:]


def symmetric(idx, val):
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


def randomwalk(idx, val):
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

    D = L.sum(dim=1)  # .pow(-1).diag()
    L = torch.linalg.solve(D.diag(), L)  # torch.mm(D.pow(-1).diag(), L)
    # L = torch.mm(D, L)
    # L *= D.unsqueeze(-1)

    return torch.eye(dim)-L, D


def print_mat(mat):
    s = [[str(e) for e in row] for row in mat]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


# Symmetric Laplacian
Ls = symmetric(idx, val)

# mv = torch.mv(Ls, y)
# mv_solve = torch.linalg.solve(Ls, y)

# Random Walk Laplacian
Lr, D = randomwalk(idx, val)

# mv = torch.mv(Lr, y)
# mv_solve = torch.linalg.solve(Lr, y)

# Precision Symmetric
Ps = torch.eye(nodes.shape[0])*2*nu/lengthscale.square().squeeze() + Ls
Ps = torch.mm(Ps, torch.mm(Ps, Ps))

# mv = torch.mv(Ps, y)
# mv_solve = torch.linalg.solve(Ps y)

# Precision Random Walk
Pr = torch.eye(nodes.shape[0])*2*nu/lengthscale.square().squeeze() + Lr
# Dg = D.detach()
Pr = torch.mm(torch.mm(Pr, torch.mm(Pr, Pr)), D.pow(-1).diag())
# Pr = torch.mm(Pr, torch.mm(Pr, Pr))
Pr /= outputscale
Qxx = Pr[:, labeled]
Qxx = Qxx[labeled, :]

Qxz = Pr[:, not_labeled]
Qxz = Qxz[labeled, :]

Qzz = Pr[:, not_labeled]
Qzz = Qzz[not_labeled, :]

Qzx = Pr[:, labeled]
Qzx = Qzx[not_labeled, :]

# Pr = Qxx - torch.mm(Qxz, torch.linalg.solve(Qzz, Qzx))
Pr = Pr - noise*torch.mm(Pr, Pr) + noise.square()*torch.mm(Pr, torch.mm(Pr, Pr))

mv = torch.mv(Pr, y)
# mv = torch.mv(Pr, z)

# mv_solve = torch.linalg.solve(Pr, y)

# opt = kernel.base_kernel.laplacian(operator='randomwalk')

opt = model.noise_precision()
# opt = NoiseWrapper(noise, ScaleWrapper(outputscale, SemiSupervisedWrapper(labeled, kernel.base_kernel.precision())))
# opt = NoiseWrapper(noise, ScaleWrapper(outputscale, kernel.base_kernel.precision()))

# mv_opt = opt.matmul(z.view(-1, 1)).squeeze()
mv_opt = opt.matmul(y.view(-1, 1)).squeeze()

# mv_solve_opt = opt.solve(y.view(-1, 1)).squeeze()


# with gpytorch.settings.max_cholesky_size(300):
#     loss = opt.inv_quad_logdet(logdet=True)[1]  # mv_opt.sum()

# loss = 0.5 * sum([torch.dot(z, mv), -torch.logdet(Pr), z.size(-1) * math.log(2 * math.pi)])  # mv.sum()
loss = 0.5 * sum([torch.dot(y, mv), -torch.logdet(Pr), y.size(-1) * math.log(2 * math.pi)])  # mv.sum()

# with gpytorch.settings.max_cholesky_size(300):
# loss = 0.5 * sum([torch.dot(z, opt.matmul(z.unsqueeze(-1)).squeeze()), -opt.inv_quad_logdet(logdet=True)[1], z.size(-1) * math.log(2 * math.pi)])
# loss = 0.5 * sum([torch.dot(y, opt.matmul(y.unsqueeze(-1)).squeeze()), -opt.inv_quad_logdet(logdet=True)[1], y.size(-1) * math.log(2 * math.pi)])

# D.retain_grad()
loss.backward()

print(loss)
print(kernel.base_kernel.raw_epsilon.grad)
print(kernel.base_kernel.raw_lengthscale.grad)
print(kernel.raw_outputscale.grad)
print(likelihood.raw_noise.grad)


# with gpytorch.settings.max_cholesky_size(300):
#     loss = opt.inv_quad_logdet(logdet=True)[1]
# loss.backward()
