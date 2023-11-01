#!/usr/bin/env python
# encoding: utf-8

import torch
import gpytorch

from .riemann_kernel import RiemannKernel, LaplacianRandomWalk

from typing import Callable, Optional, Tuple
from linear_operator import LinearOperator, to_linear_operator, settings, utils


class RiemannMaternKernel(RiemannKernel):
    has_lengthscale = True

    def __init__(self, nu: Optional[int] = 2, **kwargs):
        super(RiemannMaternKernel, self).__init__(**kwargs)
        self.nu = torch.tensor(nu, dtype=torch.float)

    def spectral_density(self):
        s = (2*self.nu / self.lengthscale.square() + self.eigenvalues).pow(-self.nu)  # *self.nodes.shape[0]
        # s /= s.sum()
        # s *= self.nodes.shape[0]
        # s = s.sqrt()
        return s

    def precision(self):
        return PrecisionMatern(self.nu, self.lengthscale, self.laplacian(operator=self.operator))


class PrecisionMatern(LinearOperator):
    def __init__(self, nu, lengthscale, laplacian):
        super(PrecisionMatern, self).__init__(nu, lengthscale, laplacian)

    # def _matmul(self, x):
    #     if self._args[2].__class__.__name__ == "LaplacianRandomWalk":
    #         res = x.mul(self._args[2]._args[1].view(-1, 1))
    #     else:
    #         res = x

    #     diagonal_term = 2*self._args[0] / self._args[1].square().squeeze()

    #     for _ in range(self._args[0].int()):
    #         res = res * diagonal_term + self._args[2]._matmul(res)

    #     return res

    def _matmul(self, x):
        res = x

        diagonal_term = self._args[1].square().squeeze() / (2*self._args[0])

        for _ in range(self._args[0].int()):
            res = res + diagonal_term * self._args[2]._matmul(res)
            res /= diagonal_term

        if self._args[2].__class__.__name__ == "LaplacianRandomWalk":
            res.mul_(self._args[2]._args[1].pow(-1).view(-1, 1))

        return res  # * diagonal_term.pow(-self._args[0])

    # def _solve(self, rhs: torch.Tensor, preconditioner: Callable, num_tridiag: int = 0) -> torch.Tensor:
    #     res = rhs

    #     def matmul(rhs):
    #         diagonal_term = 2*self._args[0] / self._args[1].square().squeeze()
    #         return rhs * diagonal_term + self._args[2]._matmul(rhs)

    #     for _ in range(self._args[0].int()):
    #         res = utils.linear_cg(matmul, res, n_tridiag=num_tridiag,
    #                               max_iter=settings.max_cg_iterations.value(),
    #                               max_tridiag_iter=settings.max_lanczos_quadrature_iterations.value(),
    #                               preconditioner=preconditioner,
    #                               )
    #     if self._args[2].__class__.__name__ == "LaplacianRandomWalk":
    #         res.mul_(self._args[2]._args[1].pow(-1).view(-1, 1))

    #     return res

    def _average_variance(self, num_rand_vec=None):
        if num_rand_vec is not None:
            num_points = self._size()[0]
            rand_idx = torch.randint(0, num_points-1, (1, num_rand_vec))
            rand_vec = torch.zeros(num_points, num_rand_vec).scatter_(0, rand_idx, 1.0).to(self._args[1].device)
        else:
            num_rand_vec = self._size()[0]
            rand_vec = torch.eye(self._size()[0]).to(self._args[1].device)

        with gpytorch.settings.max_cholesky_size(1):
            norm_var = self.inv_quad_logdet(inv_quad_rhs=rand_vec, logdet=False)[0]/num_rand_vec

        return norm_var

    def _size(self):
        return self._args[2]._size()

    def _transpose_nonbatch(self):
        return self
