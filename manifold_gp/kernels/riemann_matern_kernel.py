#!/usr/bin/env python
# encoding: utf-8

from typing import Optional

from .riemann_kernel import RiemannKernel
from ..operators import PrecisionMaternOperator


class RiemannMaternKernel(RiemannKernel):
    has_lengthscale = True

    def __init__(
        self,
        nu: Optional[int] = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nu = nu

    def spectral_density(self):
        return (2*self.nu / self.lengthscale.square() + self.eigval).pow(-self.nu)

    def precision(self):
        return PrecisionMaternOperator(self.laplacian(), self.nu, self.lengthscale)
