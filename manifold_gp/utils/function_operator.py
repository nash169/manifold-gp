#!/usr/bin/env python
# encoding: utf-8

import torch
from linear_operator import LinearOperator
from typing import Union


class FunctionOperator(LinearOperator):
    def __init__(self, x, function):
        super(FunctionOperator, self).__init__(x, function=function)

    def _matmul(self, x):
        return self._kwargs['function'](x)

    def _size(self):
        return torch.Size([self._args[0].shape[0], self._args[0].shape[0]])

    def _transpose_nonbatch(self):
        return self

    def evaluate_kernel(self):
        return self

    def matmul(self, other: Union[torch.Tensor, "LinearOperator"]) -> Union[torch.Tensor, "LinearOperator"]:
        return self._matmul(other)
