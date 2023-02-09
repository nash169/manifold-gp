#!/usr/bin/env python
# encoding: utf-8

import torch
from typing import Optional
from .riemann_kernel import RiemannKernel


class RiemannMaternKernel(RiemannKernel):
    has_lengthscale = True

    def __init__(self, nu: Optional[int] = 2, **kwargs):
        super(RiemannMaternKernel, self).__init__(**kwargs)
        self.nu = nu

    def spectral_density(self):
        s = (2*self.nu / self.lengthscale.square() +
             self.eigenvalues).pow(-self.nu)  # *self.nodes.shape[0]
        return s / s.sum()

    # def base_precision_matrix(self):
    #     # Similarity matrix
    #     val = self.values.div(-2*self.epsilon.square()).exp()

    #     # Degree matrix
    #     deg = val.sum(dim=1)

    #     # Symmetric normalization
    #     val = val.div(deg.sqrt().unsqueeze(1)).div(
    #         deg.sqrt()[self.indices[:, 1:]])

    #     # Base Precision matrix
    #     return torch.cat((torch.ones(self.nodes.shape[0], 1).to(self.nodes.device) + 2*self.nu/self.lengthscale.square(), -val), dim=1)

    def base_precision_matrix(self):
        # Similarity matrix
        val = self.laplacian(alpha=self.alpha, type=self.ltype)
        val[:, 0] += 2*self.nu/self.lengthscale.square().squeeze()

        return val
