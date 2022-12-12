#!/usr/bin/env python
# encoding: utf-8

import math
import torch
import gpytorch

from src.utils.sparse_operator import SparseOperator
from src.utils.function_operator import FunctionOperator


class RiemannGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, labels=None):
        super().__init__(train_x, train_y, likelihood)

        # This models have constant mean
        self.mean_module = gpytorch.means.ConstantMean()

        # Riemann kernel (add check for it)
        self.covar_module = kernel

        # Store labels in case of semi-supervised scenario
        self.labels = labels

        # Preset parameters
        self.likelihood.noise_covar.raw_noise = torch.nn.Parameter(
            torch.tensor(-5.0), requires_grad=True)
        if hasattr(self.covar_module, 'base_kernel'):
            self.covar_module.raw_outputscale = torch.nn.Parameter(
                torch.tensor(-1.0), requires_grad=True)
            self.covar_module.base_kernel.raw_lengthscale = torch.nn.Parameter(
                torch.tensor(-1.0), requires_grad=True)
        else:
            self.covar_module.raw_lengthscale = torch.nn.Parameter(
                torch.tensor(-1.0), requires_grad=True)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def noiseless_precision_matrix(self, x):
        kernel = self.covar_module.base_kernel if hasattr(
            self.covar_module, 'base_kernel') else self.covar_module

        # Base Precision Matrix
        val = kernel.base_precision_matrix()

        if self.labels is not None:
            y = torch.zeros(kernel.nodes.shape[0], x.shape[1]).to(x.device)
            y[self.labels, :] = x

            for _ in range(kernel.nu):
                y = torch.sum(
                    val * y[kernel.indices].permute(2, 0, 1), dim=2).t()

            Q_xx = y[self.labels, :]
            opt = SparseOperator(val, kernel.indices, torch.Size(
                [kernel.nodes.shape[0], kernel.nodes.shape[0]]))
            y[self.labels, :] = 0.0

            for _ in range(kernel.nu):
                y = opt.solve(y)

            not_labaled = torch.ones_like(y)
            not_labaled[self.labels, :] = 0.0
            z = y*not_labaled

            for _ in range(kernel.nu):
                z = torch.sum(
                    val * z[kernel.indices].permute(2, 0, 1), dim=2).t()

            z = Q_xx + z[self.labels, :]

            if hasattr(self.covar_module, 'outputscale'):
                z /= self.covar_module.outputscale**2

            return z
        else:
            y = x

            for _ in range(kernel.nu):
                y = torch.sum(
                    val * y[kernel.indices].permute(2, 0, 1), dim=2).t()

            if hasattr(self.covar_module, 'outputscale'):
                y /= self.covar_module.outputscale**2

            return y

    def noise_precision_matrix(self, x):
        return self.noiseless_precision_matrix(x - self.likelihood.noise.pow(2)*self.noiseless_precision_matrix(x + self.likelihood.noise.pow(4)*self.noiseless_precision_matrix(x)))

    def manifold_informed_train(self, lr=1e-1, iter=100, verbose=True):
        # Training targets
        y = self.train_targets

        # Deactivate optimization mean parameters
        self.mean_module.raw_constant = torch.nn.Parameter(
            self.mean_module.raw_constant.data, requires_grad=False)

        # Activate optimization of the Laplacian lengthscale
        try:
            self.covar_module.raw_epsilon = torch.nn.Parameter(
                self.covar_module.raw_epsilon.data, requires_grad=True)
        except:
            self.covar_module.base_kernel.raw_epsilon = torch.nn.Parameter(
                self.covar_module.base_kernel.raw_epsilon.data, requires_grad=True)

        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Noise Precision Matrix Operator
        operator = FunctionOperator(y, self.noise_precision_matrix)

        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.item():0.3f} ", end='')

        with gpytorch.settings.fast_computations(log_prob=False) and gpytorch.settings.max_cholesky_size(300) and torch.autograd.set_detect_anomaly(True):
            for i in range(iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()

                output = self.noise_precision_matrix(y)
                opt = FunctionOperator(y, self.noise_precision_matrix)

                loss = 0.5 * sum([torch.dot(y.squeeze(), output.squeeze()),
                                  -opt.inv_quad_logdet(logdet=True)[1], y.size(-1) * math.log(2 * math.pi)])

                loss.backward()

                # Print step information
                if verbose:
                    print(
                        f"Iteration: {i}, Loss: {loss.item():0.3f}, Noise Variance: {self.likelihood.noise.item():0.3f}", end='')
                    if hasattr(self.covar_module, 'outputscale'):
                        print(
                            f", Signal Variance: {self.covar_module.outputscale.item():0.3f}", end='')
                    if hasattr(self.covar_module, 'base_kernel'):
                        print(
                            f", Lengthscale: {self.covar_module.base_kernel.lengthscale.item():0.3f}, Epsilon: {self.covar_module.base_kernel.epsilon.item():0.3f}")
                    else:
                        print(
                            f", Lengthscale: {self.covar_module.lengthscale.item():0.3f}, Epsilon: {self.covar_module.epsilon.item():0.3f}")

                # Step
                optimizer.step()
