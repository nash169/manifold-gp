#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from scipy.sparse import coo_matrix


class LaplacianJax():
    def __init__(self, indices, distances):
        super(LaplacianJax, self).__init__()

        # # Laplacian heat kernel hyperparameter
        # self.eps_ = nn.Parameter(
        #     torch.tensor(0.01), requires_grad=True)

        row = np.arange(indices.shape[0]).repeat(indices.shape[1])
        col = indices.reshape(1, -1)
        data = -distances.reshape(1, -1)

        self.L_ = coo_matrix((data, (row, col)), shape=(
            indices.shape[0], indices.shape[0]))

        # # Covariance function length hyperparameter
        # self.k_ = nn.Parameter(torch.tensor(0.1), requires_grad=True)

        # # Covariance function signal variance hyperparameter
        # self.sigma_ = nn.Parameter(torch.tensor(1.), requires_grad=True)

        # # Covariance function noise variance hyperparameter
        # self.sigma_n_ = nn.Parameter(torch.tensor(1e-3), requires_grad=True)

    # def forward(self, x):

    #     return
