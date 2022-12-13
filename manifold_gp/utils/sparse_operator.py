#!/usr/bin/env python
# encoding: utf-8

import torch
from linear_operator import LinearOperator
from typing import Union


class SparseOperator(LinearOperator):
    def __init__(self, values, indices, size):
        super(SparseOperator, self).__init__(values,indices,size=size)

    def _size(self):
        return self._kwargs['size']

    def _matmul(self, x):
        return torch.sum(self._args[0] * x[self._args[1]].permute(2, 0, 1), dim=2).t()

    def _transpose_nonbatch(self):
        return self

    def evaluate_kernel(self):
        return self

    def matmul(self, other: Union[torch.Tensor, "LinearOperator"]) -> Union[torch.Tensor, "LinearOperator"]:
        return self._matmul(other)