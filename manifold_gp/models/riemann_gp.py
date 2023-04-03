#!/usr/bin/env python
# encoding: utf-8

import math
import torch
import gpytorch

from linear_operator import LinearOperator
from typing import Union


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

    def eval(self):
        # Generate eigenfunctions for kernel evaluation
        if hasattr(self.covar_module, 'base_kernel'):
            self.covar_module.base_kernel.generate_eigenpairs()
        else:
            self.covar_module.generate_eigenpairs()

        return super().eval()

    def noise_precision(self):
        # Supervised / Semisupervised Learning
        if self.labels is not None:
            opt = SemiSupervisedWrapper(self.labels, self.covar_module.base_kernel.precision() if hasattr(self.covar_module, 'base_kernel') else self.covar_module.precision())
        else:
            opt = self.covar_module.base_kernel.precision() if hasattr(self.covar_module, 'base_kernel') else self.covar_module.precision()

        # Scale output
        if hasattr(self.covar_module, 'outputscale'):
            opt = ScaleWrapper(self.covar_module.outputscale, opt)

        return NoiseWrapper(self.likelihood.noise, opt)

    def manifold_informed_train(self, lr=1e-1, iter=100, verbose=True):
        self.train()
        self.likelihood.train()

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
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-8)

        # Iterations
        for i in range(iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()

            # Operator
            opt = self.noise_precision()

            # Loss
            loss = 0.5 * sum([torch.dot(y.squeeze(), opt.matmul(y).squeeze()), -opt.inv_quad_logdet(logdet=True)[1], y.size(-1) * math.log(2 * math.pi)])

            # Add log probs of priors on the (functions of) parameters
            loss_ndim = loss.ndim
            for _, module, prior, closure, _ in self.named_priors():
                prior_term = prior.log_prob(closure(module))
                loss.add_(prior_term.view(*prior_term.shape[:loss_ndim], -1).sum(dim=-1))

            # Gradient
            loss.backward()

            # Print step information
            if verbose:
                print(f"Iteration: {i}, Loss: {loss.item():0.3f}, Noise Variance: {self.likelihood.noise.sqrt().item():0.3f}", end='')
                if hasattr(self.covar_module, 'outputscale'):
                    print(f", Signal Variance: {self.covar_module.outputscale.sqrt().item():0.3f}", end='')
                if hasattr(self.covar_module, 'base_kernel'):
                    print(f", Lengthscale: {self.covar_module.base_kernel.lengthscale.item():0.3f}, Epsilon: {self.covar_module.base_kernel.epsilon.item():0.3f}")
                else:
                    print(f", Lengthscale: {self.covar_module.lengthscale.item():0.3f}, Epsilon: {self.covar_module.epsilon.item():0.3f}")

            # Step
            optimizer.step()

        # Activate optimization mean parameters
        self.mean_module.raw_constant.requires_grad = True

        # Deactivate optimization of the Laplacian lengthscale
        try:
            self.covar_module.raw_epsilon.requires_grad = False
        except:
            self.covar_module.base_kernel.raw_epsilon.requires_grad = False


class ScaleWrapper(LinearOperator):
    def __init__(self, outputscale, operator):
        super(ScaleWrapper, self).__init__(outputscale, operator)

    def _matmul(self, x):
        return self._args[1]._matmul(x)/self._args[0]

    def _size(self):
        return self._args[1]._size()

    def _transpose_nonbatch(self):
        return self


class NoiseWrapper(LinearOperator):
    def __init__(self, noise, operator):
        super(NoiseWrapper, self).__init__(noise, operator)

    def _matmul(self, x):
        return self._args[1]._matmul(x - self._args[0]*self._args[1]._matmul(x - self._args[0]*self._args[1]._matmul(x)))

    def _size(self):
        return self._args[1]._size()

    def _transpose_nonbatch(self):
        return self


class SemiSupervisedWrapper(LinearOperator):
    def __init__(self, indices, operator):
        super(SemiSupervisedWrapper, self).__init__(indices, operator)

    def _matmul(self, x):
        y = torch.zeros(self._args[1]._size()[0], x.shape[1]).to(x.device)
        y[self._args[0], :] = x
        y = self._args[1]._matmul(y)

        Q_xx = y[self._args[0], :]

        y[self._args[0], :] = 0.0
        y = self._args[1].solve(y)

        not_labaled = torch.ones_like(y)
        not_labaled[self._args[0], :] = 0.0
        z = y*not_labaled
        z = self._args[1]._matmul(z)

        return Q_xx + z[self._args[0], :]

    def _size(self):
        return torch.Size([self._args[0].shape[0], self._args[0].shape[0]])

    def _transpose_nonbatch(self):
        return self
