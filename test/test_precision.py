# !/usr/bin/env python
# encoding: utf-8

import sys
import math
import torch
import gpytorch

from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
from manifold_gp.models.riemann_gp import RiemannGP

from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.operators.precision_matern_operator import PrecisionMaternOperator
from manifold_gp.operators.scale_wrapper_operator import ScaleWrapperOperator
from manifold_gp.operators.noise_wrapper_operator import NoiseWrapperOperator
from manifold_gp.operators.schur_complement_operator import SchurComplementOperator

from linear_operator.operators import MaskedLinearOperator
from manifold_gp.models.riemann_gp import SubBlockOperator

from manifold_gp.utils.nearest_neighbors import knngraph, NearestNeighbors
from manifold_gp.utils.load_dataset import rmnist_dataset, manifold_1D_dataset
from test._dense_operators import *
from test._test_functions import *


if __name__ == "__main__":
    normalization = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ["symmetric", "randomwalk"] else "symmetric"
    semisupervised = True if len(sys.argv) > 2 and sys.argv[2] == 'semi' else False

    # sampled_x, _, _, _ = rmnist_dataset()
    # sampled_x = sampled_x.flatten(start_dim=1)[:5000]
    sampled_x, sampled_y = manifold_1D_dataset()
    (m, n) = sampled_x.shape
    torch.manual_seed(1337)
    num_labeled = 10
    idx_labeled = torch.randperm(sampled_x.shape[0])[:num_labeled]
    labeled = torch.zeros(m).scatter_(0, idx_labeled, 1).bool()
    train_x, train_y = sampled_x[labeled], sampled_y[labeled]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    sampled_x = sampled_x.contiguous().to(device)
    train_x, train_y = train_x.contiguous().to(device), train_y.contiguous().to(device)

    nu = 2
    neighbors = 5
    kernel = gpytorch.kernels.ScaleKernel(
        RiemannMaternKernel(
            nu=nu,
            nodes=sampled_x if semisupervised else train_x,
            neighbors=neighbors,
            operator=normalization,
            method="exact",
            modes=100,
            ball_scale=3.0,
            prior_bandwidth=False,
        )
    )
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-8))
    model = RiemannGP(train_x, train_y, likelihood, kernel, labeled if semisupervised else None).to(device)
    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(1.3e-2),
        'covar_module.base_kernel.epsilon': torch.tensor(0.7),
        'covar_module.base_kernel.lengthscale': torch.tensor(5.3),
        'covar_module.outputscale': torch.tensor(1.4),
    }
    model.initialize(**hypers)

    knn = NearestNeighbors(sampled_x if semisupervised else train_x, nlist=1)
    idx, val = knn.graph(kernel.base_kernel.neighbors, nprobe=1)

    laplacian_dense, _, degree_dense = graph_laplacian(idx, val, kernel.base_kernel.epsilon, m if semisupervised else num_labeled, normalization=normalization)
    precision_dense = matern_precision(laplacian_dense, nu, kernel.base_kernel.lengthscale, degree_dense if normalization == 'randomwalk' else None)
    if semisupervised:
        precision_dense = matern_labeled_precision(precision_dense, labeled)
    precision_scaled_dense = matern_scaled_precision(precision_dense, kernel.outputscale)
    precision_noisy_dense = matern_noisy_precision(precision_scaled_dense, likelihood.noise)

    laplacian_sparse = GraphLaplacianOperator(val, idx, m if semisupervised else num_labeled, kernel.base_kernel.epsilon, normalization)  # kernel.laplacian(normalization)
    precision_sparse = PrecisionMaternOperator(laplacian_sparse, nu, kernel.base_kernel.lengthscale)
    if semisupervised:
        precision_sparse = SchurComplementOperator(precision_sparse, labeled)
    precision_scaled_sparse = ScaleWrapperOperator(precision_sparse, kernel.outputscale, inverse_scale=True)
    precision_noisy_sparse = NoiseWrapperOperator(precision_scaled_sparse, likelihood.noise)

    # matern precision
    v = torch.rand(num_labeled).to(device)
    with gpytorch.settings.max_cholesky_size(2000):
        test_mv(precision_dense, precision_sparse, v)
    # test_mv_transpose(laplacian_dense, laplacian_sparse, v)

    # test precision loss
    # test_ml(precision_dense, precision_sparse, train_y, max_cholesky=2000, cg_tolerance=1e-4, cg_max_iter=1000,
    #         graphbandwidth=kernel.base_kernel.raw_epsilon,
    #         lengthscale=kernel.base_kernel.raw_lengthscale)

    # test scaled precision loss
    # test_ml(precision_scaled_dense, precision_scaled_sparse, train_y, max_cholesky=800, cg_tolerance=1e-2, cg_max_iter=1000,
    #         graphbandwidth=kernel.base_kernel.raw_epsilon,
    #         lengthscale=kernel.base_kernel.raw_lengthscale,
    #         outputscale=kernel.raw_outputscale)

    # test noisy precision loss
    test_ml(precision_noisy_dense, precision_noisy_sparse, train_y, max_cholesky=2000, cg_tolerance=1e-2, cg_max_iter=1000,
            graphbandwidth=kernel.base_kernel.raw_epsilon,
            lengthscale=kernel.base_kernel.raw_lengthscale,
            outputscale=kernel.raw_outputscale,
            noise=likelihood.raw_noise)
