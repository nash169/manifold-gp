#!/usr/bin/env python
# encoding: utf-8

import torch

from .riemann_kernel import RiemannKernel, LaplacianRandomWalk

from typing import Optional, Tuple
from linear_operator import LinearOperator, to_linear_operator


class RiemannMaternKernel(RiemannKernel):
    has_lengthscale = True

    def __init__(self, nu: Optional[int] = 2, **kwargs):
        super(RiemannMaternKernel, self).__init__(**kwargs)
        self.nu = torch.tensor(nu, dtype=torch.float)

    def spectral_density(self):
        s = (2*self.nu / self.lengthscale.square() + self.eigenvalues).pow(-self.nu)
        return s / s.sum()

    def precision(self):
        return PrecisionMatern(self.nu, self.lengthscale, self.laplacian(operator='randomwalk'))


class PrecisionMatern(LinearOperator):
    def __init__(self, nu, lengthscale, laplacian):
        super(PrecisionMatern, self).__init__(nu, lengthscale, laplacian)

    def _matmul(self, x):
        x = self._args[2]._args[1].view(-1, 1) * x

        for _ in range(self._args[0].int()):
            x = x * 2*self._args[0] / self._args[1].square().squeeze() + self._args[2]._matmul(x)

        return x

    def _size(self):
        return self._args[2]._size()

    def _transpose_nonbatch(self):
        return self
