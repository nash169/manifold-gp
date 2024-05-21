#!/usr/bin/env python
# encoding: utf-8

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

    def posterior(self, x, noisy_posterior=False):
        self.model_posterior = self.likelihood(self(x)) if noisy_posterior else self(x)
        return self

    @property
    def base_kernel(self):
        return self.covar_module.base_kernel if hasattr(self.covar_module, 'base_kernel') else self.covar_module

    @property
    def posterior_mean(self):
        return self.model_posterior.mean

    @property
    def posterior_covar(self):
        return self.model_posterior.lazy_covariance_matrix.evaluate_kernel()

    @property
    def posterior_stddev(self):
        return self.model_posterior.stddev
