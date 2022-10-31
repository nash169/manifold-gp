#!/usr/bin/env python
# encoding: utf-8

from pickletools import read_unicodestring1
import torch
import torch.nn as nn


class LaplacianKnn(nn.Module):
    def __init__(self, indices, distances, nu=1):
        super(LaplacianKnn, self).__init__()

        # Laplacian heat kernel hyperparameter
        self.eps_ = nn.Parameter(
            torch.tensor(0.01), requires_grad=True)

        # Covariance function length hyperparameter
        self.k_ = nn.Parameter(torch.tensor(1.), requires_grad=True)

        # Covariance function signal variance hyperparameter
        self.sigma_ = nn.Parameter(torch.tensor(1.2), requires_grad=True)

        # Covariance function noise variance hyperparameter
        self.sigma_n_ = nn.Parameter(torch.tensor(1e-3), requires_grad=False)

        # Regularity parameter
        self.nu_ = nu

        # Laplace-Beltrami Operator
        self.values_ = distances.div(-self.eps_).exp()
        D = self.values_.sum(dim=1)
        self.values_ = self.values_.div(D.unsqueeze(1)).div(D[indices])
        self.values_ = torch.cat(
            (torch.ones(indices.shape[0], 1), -self.values_), dim=1)
        self.indices_ = torch.cat(
            (torch.arange(indices.shape[0]).unsqueeze(1), indices), dim=1)

        # Precision matrix
        self.values_[:, 0] += 2*self.nu_/self.k_**2

    def forward(self):
        rows = torch.arange(self.indices_.shape[0]).repeat_interleave(
            self.indices_.shape[1]).unsqueeze(0)
        cols = self.indices_.reshape(1, -1)
        values = self.values_.reshape(1, -1).squeeze()
        return torch.sparse_coo_tensor(
            torch.cat((rows, cols), dim=0), values, (self.indices_.shape[0], self.indices_.shape[0]))

    def log_likelihood(self, y):
        return torch.dot(y, torch.sum(self.values_*y[self.indices_], dim=1))
