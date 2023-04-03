#!/usr/bin/env python
# encoding: utf-8

import torch
import gpytorch


class VanillaGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(VanillaGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def vanilla_train(self, lr=1e-1, iter=100, verbose=True):
        self.train()
        self.likelihood.train()

        # Deactivate optimization mean parameters
        self.mean_module.raw_constant.requires_grad = False

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(iter):
            optimizer.zero_grad()
            output = self(self.train_inputs[0])
            loss = -mll(output, self.train_targets)
            loss.backward()

            if verbose:
                print(f"Iteration: {i}, Loss: {loss.item():0.3f}, Noise Variance: {self.likelihood.noise.sqrt().item():0.3f}", end='')
                if hasattr(self.covar_module, 'outputscale'):
                    print(f", Signal Variance: {self.covar_module.outputscale.sqrt().item():0.3f}", end='')
                if hasattr(self.covar_module, 'base_kernel'):
                    print(f", Lengthscale: {self.covar_module.base_kernel.lengthscale.item():0.3f}")
                else:
                    print(f", Lengthscale: {self.covar_module.lengthscale.item():0.3f}")

            optimizer.step()

        # Activate optimization mean parameters
        self.mean_module.raw_constant.requires_grad = True
