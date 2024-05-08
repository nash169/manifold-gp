#!/usr/bin/env python
# encoding: utf-8

import torch
import gpytorch

from linear_operator.utils.memoize import cached
from ..operators import SchurComplementOperator, ScaleWrapperOperator, NoiseWrapperOperator
from ..utils import bump_function


class RiemannGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, labeled=None):
        super().__init__(
            train_x,
            train_y,
            likelihood
        )

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        self.labeled = labeled

    def eval(self):
        self.base_kernel.eval()
        return super().eval()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def precision(self, noise=True):
        opt = SchurComplementOperator(self.base_kernel.precision(), self.labeled) if self.labeled is not None else self.base_kernel.precision()
        if hasattr(self.covar_module, 'outputscale'):
            opt = ScaleWrapperOperator(opt, self.covar_module.outputscale)
        if noise:
            opt = NoiseWrapperOperator(opt, self.likelihood.noise)

        return opt

    def modulation(self, x):
        edge_value, _ = self.base_kernel.knn.search(x, 1)
        return bump_function(edge_value.sqrt().squeeze(), self.base_kernel.bump_scale*self.base_kernel.graphbandwidth.squeeze(), self.base_kernel.bump_decay)

    def posterior(self, x, noisy_posterior=False, base_model=None):
        with torch.no_grad():
            self.posterior_geom = self.likelihood(self(x)) if noisy_posterior else self(x)
            if base_model is not None:
                self.posterior_base = base_model.likelihood(base_model(x)) if noisy_posterior else base_model(x)
                self.base_scale = 1 - self.modulation(x)
        return self

    @property
    def base_kernel(self):
        return self.covar_module.base_kernel if hasattr(self.covar_module, 'base_kernel') else self.covar_module

    @property
    def mean(self):
        return self.posterior_geom.mean + self.base_scale*self.posterior_base.mean if hasattr(self, "posterior_base") else self.posterior_geom.mean

    @property
    def stddev(self):
        return self.posterior_geom.stddev + self.base_scale*self.posterior_base.stddev if hasattr(self, "posterior_base") else self.posterior_geom.stddev
