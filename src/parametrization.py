#!/usr/bin/env python
# encoding: utf-8

from numpy import indices
import torch
import torch.nn as nn


class MaternCovariance(nn.Module):
    def __init__(self, indices, distances):
        super(MaternCovariance, self).__init__()

        # Laplacian heat kernel hyperparameter
        self.eps_ = nn.Parameter(
            torch.tensor(0.01), requires_grad=True)
        self.L_ = torch.sparse_coo_tensor(
            torch.cat((torch.arange(indices.shape[0]).repeat_interleave(indices.shape[1]).unsqueeze(0), indices.reshape(1, -1)), dim=0), distances.div(-self.eps_).exp().reshape(1, -1).squeeze(), (indices.shape[0], indices.shape[0]))

        # Covariance function length hyperparameter
        self.k_ = nn.Parameter(torch.tensor(0.1), requires_grad=True)

        # Covariance function signal variance hyperparameter
        self.sigma_ = nn.Parameter(torch.tensor(1.), requires_grad=True)

        # Covariance function noise variance hyperparameter
        self.sigma_n_ = nn.Parameter(torch.tensor(1e-3), requires_grad=True)

    def forward(self, x):

        return
