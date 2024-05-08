#!/usr/bin/env python
# encoding: utf-8

from typing import Optional

from .riemann_kernel_tmp import RiemannKernelTmp
from ..operators import PrecisionMaternOperator


class RiemannMaternKernelTmp(RiemannKernelTmp):
    has_lengthscale = True

    def __init__(
        self,
        nu: Optional[int] = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nu = nu

    def spectral_density(self):
        return (2*self.nu / self.lengthscale.square() + self.eigenvalues).pow(-self.nu)

    def precision(self):
        return PrecisionMaternOperator(self.laplacian(operator=self.operator), self.nu, self.lengthscale)
