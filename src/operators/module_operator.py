#!/usr/bin/env python
# encoding: utf-8

import torch
from linear_operator import LinearOperator
from typing import Union


class ModuleOperator(LinearOperator):
    def __init__(self, X, module):
        super(ModuleOperator, self).__init__(X, module=module)

        self.X_ = X
        self.module_ = module

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
