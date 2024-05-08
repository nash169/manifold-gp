# !/usr/bin/env python
# encoding: utf-8

import time
import torch
import sys

from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel

from manifold_gp.utils.nearest_neighbors import knngraph
from manifold_gp.utils.load_dataset import rmnist_dataset
from test._dense_operators import graph_laplacian


def bench_dense_mv(dense_operator, vector):
    print("Benchmark matrix-vector multiplication")
    t0 = time.time()
    torch.mm(dense_operator, vector.view(-1, 1))
    print("Dense Operator: ", time.time()-t0)


def bench_dense_solve(dense_operator, vector):
    pass


def bench_dense_grad(dense_operator, vector):
    print("Test gradient")
    loss = torch.mm(dense_operator, vector.view(-1, 1)).sum()
    t0 = time.time()
    loss.backward()
    print("Dense Time: ", time.time()-t0)


def bench_dense_eigen(dense_operator):
    print("Benchmark Eigen")
    t0 = time.time()
    torch.linalg.eigh(dense_operator)
    print("Dense Time: ", time.time()-t0)


if __name__ == "__main__":
    sampled_x, _, _, _ = rmnist_dataset()
    sampled_x = sampled_x.flatten(start_dim=1)[:5000]
    # sampled_x, _ = manifold_1D_dataset()
    dim = sampled_x.shape[0]

    # kernel
    kernel = RiemannMaternKernel(
        nu=1,
        nodes=sampled_x,
        neighbors=50,
        operator="symmetric",
        method="exact",
        modes=100,
        ball_scale=3.0,
        prior_bandwidth=False,
    )

    # laplacian
    idx, val = knngraph(sampled_x, kernel.neighbors)
    laplacian, _, _ = graph_laplacian(idx, val, kernel.epsilon, dim, normalization="symmetric")

    # test vector
    torch.manual_seed(1337)
    v = torch.rand(dim)

    benchmark = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ["mv", "grad", "eigen"] else "mv"
    if benchmark == "mv":
        bench_dense_mv(laplacian, v)
    elif benchmark == "grad":
        bench_dense_grad(laplacian, v)
    elif benchmark == "eigen":
        bench_dense_eigen(laplacian)
