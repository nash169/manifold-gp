#!/usr/bin/env python
# encoding: utf-8

import torch

from .riemann_kernel import RiemannKernel

from typing import Optional, Union, Callable
from linear_operator import LinearOperator, settings, utils


class RiemannMaternKernel(RiemannKernel):
    has_lengthscale = True

    def __init__(self, nu: Optional[int] = 2, **kwargs):
        super(RiemannMaternKernel, self).__init__(**kwargs)
        self.nu = nu

    def spectral_density(self):
        s = (2*self.nu / self.lengthscale.square() +
             self.eigenvalues).pow(-self.nu)  # *self.nodes.shape[0]
        return s / s.sum()

    def precision(self):
        return PrecisionMaternOperator(self.laplacian(), kernel=self)


class PrecisionMaternOperator(LinearOperator):
    def __init__(self, laplacian, kernel):
        super(PrecisionMaternOperator, self).__init__(
            laplacian, kernel=kernel)

    def _matmul(self, x):
        r = self._kwargs['kernel'].indices[0, :]
        c = self._kwargs['kernel'].indices[1, :]
        v = self._args[0]
        nu = self._kwargs['kernel'].nu
        k = self._kwargs['kernel'].lengthscale

        for _ in range(nu):
            x = x.mul(2*nu/k.square().squeeze() + 1).index_add(0, r, v.view(-1, 1)
                                                               * x[c], alpha=-1).index_add(0, c, v.view(-1, 1)*x[r], alpha=-1)

        return x

    def _base(self, x):
        r = self._kwargs['kernel'].indices[0, :]
        c = self._kwargs['kernel'].indices[1, :]
        v = self._args[0]
        nu = self._kwargs['kernel'].nu
        k = self._kwargs['kernel'].lengthscale

        return x.mul(2*nu/k.square().squeeze() + 1).index_add(0, r, v.view(-1, 1) * x[c], alpha=-1).index_add(0, c, v.view(-1, 1)*x[r], alpha=-1)

    def _solve(self, rhs: torch.Tensor, preconditioner: Callable, num_tridiag: int = 0) -> torch.Tensor:
        return utils.linear_cg(
            self._base,
            rhs,
            n_tridiag=num_tridiag,
            max_iter=settings.max_cg_iterations.value(),
            max_tridiag_iter=settings.max_lanczos_quadrature_iterations.value(),
            preconditioner=preconditioner,
        )

    def solve(self, x: torch.Tensor) -> torch.Tensor:
        nu = self._kwargs['kernel'].nu

        with settings.fast_computations.solves(True) and settings.max_cholesky_size(1):
            for _ in range(nu):
                x = super().solve(x, None)

        return x

    def _size(self):
        dim = self._kwargs['kernel'].nodes.shape[0]
        return torch.Size([dim, dim])

    def _transpose_nonbatch(self):
        return self

    def evaluate_kernel(self):
        return self

    def matmul(self, other: Union[torch.Tensor, "LinearOperator"]) -> Union[torch.Tensor, "LinearOperator"]:
        return self._matmul(other)
