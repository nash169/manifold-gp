#!/usr/bin/env python
# encoding: utf-8

import torch
from linear_operator import LinearOperator
from typing import Union


class SparseOperator(LinearOperator):
    def __init__(self, X, module):
        super(SparseOperator, self).__init__(X, module=module)

        self.X_ = X
        self.module_ = module

        # super().settings.fast_computations(log_prob=True)

        # self._set_requires_grad(True)

    # @property
    # def dtype(self):
    #     return self.module_.dtype

    # @property
    # def device(self):
    #     return self.X_.device

    # @property
    # def requires_grad(self):
    #     return super().requires_grad or any(param.requires_grad for param in self.module_.parameters())

    # def _set_requires_grad(self, val):
    #     super()._set_requires_grad(val)
    #     # The behavior that differs from the base LinearOperator setter
    #     for param in self.module_.parameters():
    #         param.requires_grad_(val)

    def _matmul(self, x):
        y = x
        return self.module_(y)

    def _size(self):
        return self.module_.size_

    def _transpose_nonbatch(self):
        return self

    def evaluate_kernel(self):
        return self

    def matmul(self, other: Union[torch.Tensor, "LinearOperator"]) -> Union[torch.Tensor, "LinearOperator"]:
        return self._matmul(other)
