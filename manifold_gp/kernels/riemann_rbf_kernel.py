#!/usr/bin/env python
# encoding: utf-8

import torch

from .riemann_kernel import RiemannKernel

from typing import Optional, Union, Callable
from linear_operator import LinearOperator, settings, utils


class RiemannRBFKernel(RiemannKernel):
    has_lengthscale = True

    def __init__(self, **kwargs):
        super(RiemannRBFKernel, self).__init__(**kwargs)

    def spectral_density(self):
        s = (-0.5*self.lengthscale.square()*self.eigenvalues).exp()
        return s/s.sum()

    def precision(self):
        raise NotImplementedError(
            "Precision Operator not implemented for RBF kernel.")


class PrecisionRBFOperator(LinearOperator):
    def __init__(self, laplacian, kernel):
        super(PrecisionRBFOperator, self).__init__(
            laplacian, kernel=kernel)

    def _matmul(self, x):
        r = self._kwargs['kernel'].indices[0, :]
        c = self._kwargs['kernel'].indices[1, :]
        v = self._args[0]
        k = self._kwargs['kernel'].lengthscale

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
