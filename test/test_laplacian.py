# !/usr/bin/env python
# encoding: utf-8

import torch
import sys

from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator

from manifold_gp.utils.nearest_neighbors import knngraph, NearestNeighbors
from manifold_gp.utils.load_dataset import rmnist_dataset, manifold_1D_dataset
from test._dense_operators import graph_laplacian
from test._test_functions import *

from torch.nn.functional import normalize

if __name__ == "__main__":
    # sampled_x, _, _, _ = rmnist_dataset()
    # sampled_x = sampled_x.flatten(start_dim=1)[:5000]
    sampled_x, sampled_y, _ = manifold_1D_dataset()

    num_test = 10
    torch.manual_seed(1337)
    test_idx = torch.zeros(sampled_x.shape[0]).scatter_(0, torch.randperm(sampled_x.shape[0])[:num_test], 1).bool()
    train_x, test_x, train_y, test_y = sampled_x[~test_idx], sampled_x[test_idx], sampled_y[~test_idx], sampled_y[test_idx]
    # (n, d) = train_x.shape

    use_cuda = False  # torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_x, test_x, train_y, test_y = train_x.contiguous().to(device), test_x.contiguous().to(device), train_y.contiguous().to(device), test_y.contiguous().to(device)

    normalization = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ["symmetric", "randomwalk"] else "symmetric"

    # kernel
    kernel = RiemannMaternKernel(
        nu=1,
        x=train_x,
        nearest_neighbors=20,
        laplacian_normalization=normalization,
        num_modes=10,
        bump_scale=3.0,
        bump_decay=1.0,
        graphbandwidth_prior=None,
        graphbandwidth_constraint=None
    ).to(device)

    hypers = {
        'graphbandwidth': 0.02,
        'lengthscale': 0.5,
    }
    kernel.initialize(**hypers)

    # laplacian
    knn = NearestNeighbors(train_x, nlist=1)
    idx, val = knn.graph(kernel.nearest_neighbors, nprobe=1)
    laplacian_sparse = GraphLaplacianOperator(val, idx, train_x.shape[0], kernel.graphbandwidth, normalization)  # kernel.laplacian(normalization)
    laplacian_dense, _, degree_unorm_dense, _, degree_dense = graph_laplacian(idx, val, kernel.graphbandwidth, train_x.shape[0], normalization=normalization)

    test_mv(laplacian_dense, laplacian_sparse, train_y)
    test_mv_transpose(laplacian_dense, laplacian_sparse, train_y)
    test_diag(laplacian_dense, laplacian_sparse)
    test_grad(laplacian_dense, laplacian_sparse, train_y, kernel.raw_graphbandwidth)
    dense_eval, dense_evec, sparse_eval, sparse_evec = test_eigen(laplacian_dense, laplacian_sparse, normalization, max_cholesky=2000)
    dense_eval, dense_evec, sparse_eval, sparse_evec = dense_eval[:kernel.num_modes], dense_evec[:, :kernel.num_modes], sparse_eval[:kernel.num_modes], sparse_evec[:, :kernel.num_modes]
    # normalize(sparse_evec, p=2, dim=0)

    edge_value, edge_index = knn.search(test_x, kernel.nearest_neighbors)
    test_outofsample(edge_value, edge_index, dense_eval, dense_evec, laplacian_sparse, sparse_eval, sparse_evec,
                     degree_unorm_dense, degree_dense, kernel.graphbandwidth, kernel.lengthscale, kernel.nu, normalization)
