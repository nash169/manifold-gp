# !/usr/bin/env python
# encoding: utf-8

import matplotlib.pyplot as plt
from time import time
import math

import numpy as np
import torch
import gpytorch

from importlib.resources import files
from manifold_gp.utils.mesh_helper import groundtruth_from_samples
from manifold_gp.utils.file_read import get_data

from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.models.riemann_gp import RiemannGP, ScaleWrapper, NoiseWrapper, SemiSupervisedWrapper

import faiss
import faiss.contrib.torch_utils

from torch_geometric.utils import coalesce
from linear_operator import to_linear_operator

from manifold_gp.operators import SubBlockOperator, LaplacianRandomWalkOperator

# Set device
use_cuda = False  # torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load mesh and generate ground truth
data_path = files('manifold_gp.data').joinpath('dumbbell.msh')
data = get_data(data_path, "Nodes", "Elements")

vertices = data['Nodes'][:, 1:-1]
edges = data['Elements'][:, -2:].astype(int) - 1
truth, geodesics = groundtruth_from_samples(vertices, edges)

sampled_x = torch.from_numpy(vertices).float()
sampled_y = torch.from_numpy(truth).float()
(m, n) = sampled_x.shape


# train
torch.manual_seed(1337)
idx = torch.randperm(sampled_x.shape[0])
split = 50
labeled = idx[:split]
not_labeled = idx[split:]


x = sampled_x[labeled, :]
y = sampled_y[labeled]

z = sampled_x[not_labeled, :]
u = sampled_y[not_labeled]

# KNN Faiss
knn = faiss.IndexFlatL2(sampled_x.shape[1])

# Initialize kernel
nu = 1
neighbors = 10
operator = "randomwalk"
semisupervised = True

kernel = gpytorch.kernels.ScaleKernel(
    RiemannMaternKernel(
        nu=nu,
        nodes=sampled_x if semisupervised else x,
        neighbors=neighbors,
        modes=100,
        operator=operator
    )
)
likelihood = gpytorch.likelihoods.GaussianLikelihood(
    noise_constraint=gpytorch.constraints.GreaterThan(1e-8))
model = RiemannGP(x, y, likelihood, kernel, labeled if semisupervised else None)

hypers = {
    'likelihood.noise_covar.noise': torch.tensor(1e-2),
    'covar_module.base_kernel.epsilon': torch.tensor(0.05),
    'covar_module.base_kernel.lengthscale': torch.tensor(0.5),
    'covar_module.outputscale': torch.tensor(1.0),
    'covar_module.base_kernel.support_kernel.lengthscale': torch.tensor(0.1),
}
model.initialize(**hypers)

epsilon = kernel.base_kernel.epsilon
lengthscale = kernel.base_kernel.lengthscale
outputscale = kernel.outputscale
noise = likelihood.noise

knn.reset()
knn.train(kernel.base_kernel.nodes)
knn.add(kernel.base_kernel.nodes)
val, idx = knn.search(kernel.base_kernel.nodes, neighbors+1)
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

    val = val.div(-4*epsilon.square()).exp().squeeze()

    Lsparse = torch.sparse_coo_tensor(idx, val, (dim, dim))
    L = (Lsparse.to_dense() + Lsparse.to_dense().t())
    D = L.sum(dim=1).pow(-1).diag()
    L = torch.mm(D, torch.mm(L, D))

    D = L.sum(dim=1).pow(-0.5).diag()
    L = torch.mm(D, torch.mm(L, D))
    # L = torch.mm(D, L)
    # L *= D.unsqueeze(-1)

    return (torch.eye(dim)-L)/epsilon.square()


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

    val = val.div(-4*epsilon.square()).exp().squeeze()

    Lsparse = torch.sparse_coo_tensor(idx, val, (dim, dim))
    L = (Lsparse.to_dense() + Lsparse.to_dense().t())
    D = L.sum(dim=1).pow(-1).diag()
    L = torch.mm(D, torch.mm(L, D))

    D = L.sum(dim=1)  # .pow(-1).diag()
    L = torch.linalg.solve(D.diag(), L)  # torch.mm(D.pow(-1).diag(), L)
    # L = torch.mm(D, L)
    # L *= D.unsqueeze(-1)

    return (torch.eye(dim)-L)/epsilon.square(), D


def print_mat(mat):
    s = [[str(e) for e in row] for row in mat]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


if operator == 'symmetric':
    Ls = symmetric(idx, val)
    Br = torch.eye(kernel.base_kernel.nodes.shape[0])*2*nu/lengthscale.square().squeeze() + Ls
    Pr = Br
    if nu > 1:
        for _ in range(1, nu):
            Pr = torch.mm(Pr, Br)
    # Pr = torch.mm(Pr, torch.mm(Pr, Pr))
elif operator == 'randomwalk':
    Lr, D = randomwalk(idx, val)
    Br = torch.eye(kernel.base_kernel.nodes.shape[0])*2*nu/lengthscale.square().squeeze() + Lr
    Pr = Br
    if nu > 1:
        for _ in range(1, nu):
            Pr = torch.mm(Pr, Br)
    Pr = torch.mm(Pr, D.pow(-1).diag())
    # Pr = torch.mm(torch.mm(Pr, torch.mm(Pr, Pr)), D.pow(-1).diag())

# Semisupervised
if semisupervised:
    Qxx = Pr[:, labeled]
    Qxx = Qxx[labeled, :]
    Qxz = Pr[:, not_labeled]
    Qxz = Qxz[labeled, :]
    Qzz = Pr[:, not_labeled]
    Qzz = Qzz[not_labeled, :]
    Qzx = Pr[:, labeled]
    Qzx = Qzx[not_labeled, :]
    Pr = Qxx - torch.mm(Qxz, torch.linalg.solve(Qzz, Qzx))

# Outputscale
Pr /= outputscale

# Noise
Pr = Pr - noise*torch.mm(Pr, Pr) + noise.square()*torch.mm(Pr, torch.mm(Pr, Pr))

mv_mul = torch.mv(Pr, y)
mv_solve = torch.linalg.solve(Pr, y)

opt = model.noise_precision()

opt_mul = opt.matmul(y.view(-1, 1)).squeeze()
# opt_solve = opt.solve(y.view(-1, 1)).squeeze()

# loss = 0.5 * sum([torch.dot(y, mv_mul), -torch.logdet(Pr), y.size(-1) * math.log(2 * math.pi)])

with gpytorch.settings.max_cholesky_size(100):
    loss = 0.5 * sum([torch.dot(y, opt.matmul(y.unsqueeze(-1)).squeeze()), -opt.inv_quad_logdet(logdet=True)[1], y.size(-1) * math.log(2 * math.pi)])

loss.backward()

print(loss)
print(kernel.base_kernel.raw_epsilon.grad)
print(kernel.base_kernel.raw_lengthscale.grad)
print(kernel.raw_outputscale.grad)
print(likelihood.raw_noise.grad)


# tmp = kernel.base_kernel.laplacian(operator='randomwalk')
# lap = LaplacianRandomWalkOperator(tmp._args[0], tmp._args[1], tmp._args[3], tmp._args[2], kernel.base_kernel.nodes.shape[0], kernel.base_kernel)
# lap11 = SubBlockOperator(lap, labeled, labeled)
# lap12 = SubBlockOperator(lap, labeled, not_labeled)
# lap22 = SubBlockOperator(lap, not_labeled, not_labeled)

# Lxx = Lr[:, labeled]
# Lxx = Lxx[labeled, :]
# Lxz = Lr[:, not_labeled]
# Lxz = Lxz[labeled, :]
# Lzz = Lr[:, not_labeled]
# Lzz = Lzz[not_labeled, :]
# Lzx = Lr[:, labeled]
# Lzx = Lzx[not_labeled, :]

# def recursive_operation(A, B, C, n):  # n>=2
#     X, Y, Z = matrix_power(A, B, C)

#     if n > 2:
#         X, Y, Z = recursive_operation(X, Y, Z, n-1)

#     return X, Y, Z


# def matrix_power(A, B, C, X, Y, Z):
#     return (torch.mm(A, X) + torch.mm(B, Y.T), torch.mm(A, Y) + torch.mm(B, Z), torch.mm(B.T, Y) + torch.mm(C, Z))
