#!/usr/bin/env python
# encoding: utf-8

import math
import torch
import gpytorch

from ..utils.sparse_operator import SparseOperator
from ..utils.function_operator import FunctionOperator


class RiemannGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, labels=None):
        super().__init__(train_x, train_y, likelihood)

        # This models have constant mean
        self.mean_module = gpytorch.means.ConstantMean()

        # Riemann kernel (add check for it)
        self.covar_module = kernel

        # Store labels in case of semi-supervised scenario
        self.labels = labels

        # # Preset noise variance
        # self.likelihood.noise = math.exp(-5.0)

        # # Preset signal variance (if present) and lengthscale
        # if hasattr(self.covar_module, 'base_kernel'):
        #     self.covar_module.outputscale = math.exp(0.0)
        #     self.covar_module.base_kernel.lengthscale = math.exp(-1.0)
        #     self.covar_module.base_kernel.epsilon = math.exp(-2.0)
        # else:
        #     self.covar_module.lengthscale = math.exp(-1.0)
        #     self.covar_module.epsilon = math.exp(-2.0)

    # def train(self, mode=True):
    #     super().train(mode)

    #     # Build the Graph for the Laplacian approximation
    #     if hasattr(self.covar_module, 'base_kernel'):
    #         self.covar_module.base_kernel.generate_graph()
    #     else:
    #         self.covar_module.generate_graph()

    def eval(self):
        # Generate eigenfunctions for kernel evaluation
        if hasattr(self.covar_module, 'base_kernel'):
            self.covar_module.base_kernel.solve_laplacian()
        else:
            self.covar_module.solve_laplacian()

        return super().eval()

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
                z /= self.covar_module.outputscale

            return z
        else:
            y = x

            for _ in range(kernel.nu):
                y = torch.sum(
                    val * y[kernel.indices].permute(2, 0, 1), dim=2).t()

            if hasattr(self.covar_module, 'outputscale'):
                y /= self.covar_module.outputscale

            return y

    def noise_precision_matrix(self, x):
        return self.noiseless_precision_matrix(x - self.likelihood.noise*self.noiseless_precision_matrix(x + self.likelihood.noise*self.noiseless_precision_matrix(x)))

    def manifold_informed_train(self, lr=1e-1, iter=100, verbose=True):
        # Training targets
        y = self.train_targets.unsqueeze(-1)

        # Deactivate optimization mean parameters
        self.mean_module.raw_constant.requires_grad = False

        # Activate optimization of the Laplacian lengthscale
        try:
            self.covar_module.raw_epsilon.requires_grad = True
        except:
            self.covar_module.base_kernel.raw_epsilon.requires_grad = True

        # Optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=1e-8)

        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.item():0.3f} ", end='')

        with gpytorch.settings.fast_computations(log_prob=False) and gpytorch.settings.max_cholesky_size(300) and torch.autograd.set_detect_anomaly(True):
            for i in range(iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()

                # Operator
                opt = FunctionOperator(
                    y, self.noise_precision_matrix).requires_grad_(True)

                # Loss
                loss = 0.5 * sum([torch.dot(y.squeeze(), opt.matmul(y).squeeze()),
                                  -opt.inv_quad_logdet(logdet=True)[1], y.size(-1) * math.log(2 * math.pi)])

                # Gradient
                loss.backward()

                # Print step information
                if verbose:
                    print(
                        f"Iteration: {i}, Loss: {loss.item():0.3f}, Noise Variance: {self.likelihood.noise.sqrt().item():0.3f}", end='')
                    if hasattr(self.covar_module, 'outputscale'):
                        print(
                            f", Signal Variance: {self.covar_module.outputscale.sqrt().item():0.3f}", end='')
                    if hasattr(self.covar_module, 'base_kernel'):
                        print(
                            f", Lengthscale: {self.covar_module.base_kernel.lengthscale.item():0.3f}, Epsilon: {self.covar_module.base_kernel.epsilon.item():0.3f}")
                    else:
                        print(
                            f", Lengthscale: {self.covar_module.lengthscale.item():0.3f}, Epsilon: {self.covar_module.epsilon.item():0.3f}")

                # Step
                optimizer.step()

        # Activate optimization mean parameters
        self.mean_module.raw_constant.requires_grad = True

        # Deactivate optimization of the Laplacian lengthscale
        try:
            self.covar_module.raw_epsilon.requires_grad = False
        except:
            self.covar_module.base_kernel.raw_epsilon.requires_grad = False

        return loss

    def vanilla_train(self, lr=1e-1, iter=100, verbose=True):
        # Extract eigenvalues and eigenvectors
        if hasattr(self.covar_module, 'base_kernel'):
            self.covar_module.base_kernel.solve_laplacian()
        else:
            self.covar_module.solve_laplacian()

        # Train model
        self.train()
        self.likelihood.train()

        # Optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=1e-8)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()

            # Model output
            output = self.forward(self.train_inputs[0])

            # Loss
            loss = -mll(output, self.train_targets)

            # Gradient
            loss.backward()

            # Print step information
            if verbose:
                print(
                    f"Iteration: {i}, Loss: {loss.item():0.3f}, Noise Variance: {self.likelihood.noise.sqrt().item():0.3f}", end='')
                if hasattr(self.covar_module, 'outputscale'):
                    print(
                        f", Signal Variance: {self.covar_module.outputscale.sqrt().item():0.3f}", end='')
                if hasattr(self.covar_module, 'base_kernel'):
                    print(
                        f", Lengthscale: {self.covar_module.base_kernel.lengthscale.item():0.3f}, Epsilon: {self.covar_module.base_kernel.epsilon.item():0.3f}")
                else:
                    print(
                        f", Lengthscale: {self.covar_module.lengthscale.item():0.3f}, Epsilon: {self.covar_module.epsilon.item():0.3f}")

            # Step
            optimizer.step()

        return loss
