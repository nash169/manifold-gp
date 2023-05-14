#!/usr/bin/env python
# encoding: utf-8

import torch
from linear_operator import LinearOperator


class SubBlockOperator(LinearOperator):
    def __init__(self, operator, rows, cols):
        super(SubBlockOperator, self).__init__(operator, rows, cols)

    def _matmul(self, rhs):
        res = torch.zeros(self._args[0].size()[0], rhs.shape[1]).to(rhs.device)
        res[self._args[2]] = rhs

        res = self._args[0]._matmul(res)

        return res[self._args[1]]

    def _size(self):
        return torch.Size([self._args[1].shape[0], self._args[2].shape[0]])

    def _transpose_nonbatch(self):
        return SubBlockOperator(self._args[0], self._args[2], self._args[1])
