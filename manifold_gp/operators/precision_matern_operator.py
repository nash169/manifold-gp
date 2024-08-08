#!/usr/bin/env python
# encoding: utf-8

import torch
import gpytorch
from torch import Tensor
from linear_operator import LinearOperator


class PrecisionMaternOperator(LinearOperator):
    def __init__(
            self,
            laplacian: LinearOperator,
            nu: int,
            lengthscale: Tensor,
    ):
        super().__init__(
            laplacian,
            nu=nu,
            lengthscale=lengthscale
        )
        self.laplacian = laplacian
        self.nu = nu
        self.lengthscale = lengthscale

    def _matmul(self, rhs):
        out = rhs.contiguous()
        diag = self.lengthscale.square().squeeze() / (2*self.nu)

        for _ in range(self.nu):
            out = out + diag * self.laplacian._matmul(out)
            out /= diag

        if self.laplacian.normalization == "randomwalk":
            out.mul_(self.laplacian.degree_mat.view(-1, 1))

        return out

    def _size(self):
        return self.laplacian._size()

    def _transpose_nonbatch(self):
        return self

    def _average_variance(self, num_rand_vec=100):
        d = self.shape[0]
        if num_rand_vec >= d:
            rand_vec = torch.eye(d, device=self.lengthscale.device)
        else:
            rand_idx = torch.randint(0, d-1, (1, num_rand_vec), device=self.lengthscale.device)
            rand_vec = torch.zeros(d, num_rand_vec, device=self.lengthscale.device).scatter_(0, rand_idx, 1.0)

        return self.inv_quad_logdet(inv_quad_rhs=rand_vec, logdet=False)[0]/rand_vec.shape[1]
