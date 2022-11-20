#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

import gpytorch

from linear_operator import LinearOperator


class KernelOperator(LinearOperator):
    def __init__(self, X, kernel):
        super().__init__(X, kernel=kernel)
        if X.ndimension() == 1:
            self.X_ = X.unsqueeze(-1)  # .requires_grad_(True)
        else:
            self.X_ = X  # .requires_grad_(True)

        self.kernel = kernel

    def _matmul(self, x):
        # x = x.requires_grad_(True)
        # print(self.kernel)
        x = self.X_
        y = self.kernel(x, x)
        # print(y)
        return torch.matmul(y, x)

    def _size(self):
        return torch.Size([self.X_.shape[0], self.X_.shape[0]])

    def _transpose_nonbatch(self):
        return self

    @property
    def requires_grad(self):
        return super().requires_grad or any(param.requires_grad for param in self.kernel.parameters())

    def _set_requires_grad(self, val):
        super()._set_requires_grad(val)
        # The behavior that differs from the base LinearOperator setter
        for param in self.kernel.parameters():
            param.requires_grad_(val)


class SquaredExp(nn.Module):
    def __init__(self, l=1.):
        super(SquaredExp, self).__init__()

        self.sigma_ = nn.Parameter(torch.tensor(l), requires_grad=True)
        # self.kernel_ = gpytorch.kernels.RBFKernel()
        self._batch_shape = torch.Size([])
        self.active_dims = None

    def forward(self, x, y):
        l = -.5 / self.sigma**2
        xx = torch.einsum('ij,ij->i', x, x).unsqueeze(1)
        yy = torch.einsum('ij,ij->i', y, y).unsqueeze(0)
        k = -2 * torch.mm(x, y.T) + xx + yy
        k *= l
        return torch.exp(k)
        # return self.kernel_.forward(x, y)

    # Sigma variance
    @property
    def sigma(self):
        return self.sigma_

    @sigma.setter
    def sigma(self, value):
        self.sigma_ = nn.Parameter(value[0], requires_grad=value[1])

    # @property
    # def dtype(self):
    #     for param in self.parameters():
    #         return param.dtype
    #     return torch.get_default_dtype()

    # def num_outputs_per_input(self, x1, x2):
    #     """
    #     How many outputs are produced per input (default 1)
    #     if x1 is size `n x d` and x2 is size `m x d`, then the size of the kernel
    #     will be `(n * num_outputs_per_input) x (m * num_outputs_per_input)`
    #     Default: 1
    #     """
    #     return 1

    # @property
    # def batch_shape(self):
    #     return self._batch_shape

    # @batch_shape.setter
    # def batch_shape(self, val):
    #     self._batch_shape = val
