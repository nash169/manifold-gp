#!/usr/bin/env python
# encoding: utf-8

import torch

import gpytorch


class RiemannGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, labels=None):
        super().__init__(train_x, train_y, likelihood)

        # This models have constant mean
        self.mean_module = gpytorch.means.ConstantMean()

        # Riemann kernel (add check for it)
        self.covar_module = kernel

        # Store labels in case of semi-supervised scenario
        self.labels = labels

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def noiseless_precision_matrix(self, x):
        # Base Precision Matrix
        val = self.covar_module.base_precision_matrix()

        def matmul(x): return torch.sum(
            val * x[self.covar_module.indices].permute(2, 0, 1), dim=2).t()

        if self.labels is not None:
            y = torch.zeros(self.covar_module.nodes.shape[0], x.shape[1])
            y[self.labels, :] = x

            for _ in range(self.covar_module.nu):
                y = torch.sum(
                    val * y[self.covar_module.indices].permute(2, 0, 1), dim=2).t()

            Q_xx = y[self.labels, :]
            opt = SparseOperator(val, self.covar_module.indices, torch.Size(
                [self.covar_module.nodes.shape[0], self.covar_module.nodes.shape[0]]))
            y[self.labels, :] = 0.0

            for _ in range(self.covar_module.nu):
                y = opt.solve(y)

            not_labaled = torch.ones_like(y)
            not_labaled[self.labels, :] = 0.0
            z = y*not_labaled

            for _ in range(1, self.nu_):
                z = torch.sum(
                    val * z[self.covar_module.indices].permute(2, 0, 1), dim=2).t()

            return (Q_xx + z[self.labels, :])/self.signal**2
        else:
            y = x

            for _ in range(0, self.nu_):
                y = torch.sum(
                    val * y[self.idx_].permute(2, 0, 1), dim=2).t()  # ??? where is the output scale???

            return y
