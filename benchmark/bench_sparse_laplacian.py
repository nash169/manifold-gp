# !/usr/bin/env python
# encoding: utf-8

import time
import torch
import sys

from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.operators.graph_laplacian_operator import LaplacianSymmetricOperator

from manifold_gp.utils.nearest_neighbors import knngraph
from manifold_gp.utils.load_dataset import rmnist_dataset


def bench_sparse_mv(sparse_operator, vector):
    print("Benchmark matrix-vector multiplication")
    t0 = time.time()
    sparse_operator.matmul(vector.view(-1, 1))
    print("Sparse Operator: ", time.time()-t0)


def bench_sparse_grad(sparse_operator, vector):
    print("Benchmark gradient")
    loss = sparse_operator.matmul(vector.view(-1, 1)).sum()
    t0 = time.time()
    loss.backward()
    print("Sparse Time: ", time.time()-t0)


def bench_sparse_eigen(sparse_operator, num_modes=None, method="sym"):
    print("Benchmark Eigen")
    t0 = time.time()
    sparse_operator.diagonalization(method="symeig")
    print("Sparse Time: ", time.time()-t0)


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

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
    laplacian = LaplacianSymmetricOperator(val, idx, dim, kernel.epsilon)

    # test vector
    torch.manual_seed(1337)
    v = torch.rand(dim)

    benchmark = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ["mv", "grad", "eigen"] else "mv"
    if benchmark == "mv":
        bench_sparse_mv(laplacian, v)
    elif benchmark == "grad":
        bench_sparse_grad(laplacian, v)
    elif benchmark == "eigen":
        bench_sparse_eigen(laplacian)
