#!/usr/bin/env python
# encoding: utf-8

import torch
from linear_operator import LinearOperator


class BlockOperator(LinearOperator):
    def __init__(self, operator, rows, cols):
        super(BlockOperator, self).__init__(operator, rows, cols)

        self.mask = torch.isin(operator._kwargs['indices'][0], rows)*torch.isin(operator._kwargs['indices'][1], cols)
        self.rows = torch.where(operator._kwargs['indices'][0][self.mask].unsqueeze(-1) == rows)[1]
        self.cols = torch.where(operator._kwargs['indices'][1][self.mask].unsqueeze(-1) == cols)[1]

    def _matmul(self, rhs):
        return self._args[0]._matmul_block(rhs, self.mask, self.rows, self.cols, self.size())

    def _size(self):
        return torch.Size([self._args[1].shape[0], self._args[2].shape[0]])

    def _transpose_nonbatch(self):
        return BlockOperator(self._args[0], self._args[2], self._args[1])
