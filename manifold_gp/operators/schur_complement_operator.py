#!/usr/bin/env python
# encoding: utf-8

import torch
from linear_operator import LinearOperator
from .subblock_operator import SubBlockOperator


class SchurComplementOperator(LinearOperator):
    def __init__(self, operator, indices):
        super(SchurComplementOperator, self).__init__(operator, indices=indices)

        mask = torch.zeros(self._args[0]._size()[0], dtype=torch.bool).to(self._kwargs['indices'].device).scatter_(0, self._kwargs['indices'], 1)
        self.labeled = self._kwargs['indices']
        self.unlabeled = torch.masked_select(torch.arange(self._args[0]._size()[0]).to(self._kwargs['indices'].device), ~mask)

    def _matmul(self, x):
        y = torch.zeros(self._args[0]._size()[0], x.shape[1]).to(x.device)

        y[self.labeled, :] = x
        y = self._args[0]._matmul(y)

        Qxx_y = y[self.labeled, :]

        z = SubBlockOperator(self._args[0], self.unlabeled, self.unlabeled).solve(y[self.unlabeled, :])
        z = SubBlockOperator(self._args[0], self.labeled, self.unlabeled)._matmul(z)

        return Qxx_y - z

    def _size(self):
        return torch.Size([self._kwargs['indices'].shape[0], self._kwargs['indices'].shape[0]])

    def _transpose_nonbatch(self):
        return self
