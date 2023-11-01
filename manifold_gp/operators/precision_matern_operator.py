#!/usr/bin/env python
# encoding: utf-8

import torch
from linear_operator import LinearOperator
from .subblock_operator import SubBlockOperator


class PrecisionMaternOperator(LinearOperator):
    def __init__(self, nu, lengthscale, laplacian, indices):
        super(PrecisionMaternOperator, self).__init__(nu, lengthscale, laplacian, indices=indices)

    def _matmul(self, x):
        if self._args[2].__class__.__name__ == "LaplacianRandomWalk":
            res = x.mul(self._args[2]._args[1].view(-1, 1))
        else:
            res = x

        diagonal_term = 2*self._args[0] / self._args[1].square().squeeze()

        for _ in range(self._args[0].int()):
            res = res * diagonal_term + self._args[2]._matmul(res)

        return res

    def _matmul_block(self, rhs, rows, cols):
        if self._args[2].__class__.__name__ == "LaplacianRandomWalk":
            mask = torch.isin(self._args[2]._kwargs['indices'][0], rows)*torch.isin(self._args[2]._kwargs['indices'][1], cols)
            res = rhs.mul(self._args[2]._args[1][mask].view(-1, 1))
        else:
            res = rhs

        return self._block_power(res, rows, cols, self._args[0])

    def _matmul_base_block(self, rhs, rows, cols):
        if torch.equal(rows, cols):
            return rhs * (2*self._args[0] / self._args[1].square().squeeze()) + self._args[2]._matmul_block(rhs, rows, cols)
        else:
            return self._args[2]._matmul_block(rhs, rows, cols)

    def _block_power(self, rhs, rows, cols, n):
        if (n == 1):
            return self._matmul_base_block(rhs, rows, cols)
        else:
            if torch.equal(rows, cols):
                return self._block_power(self._matmul_base_block(rhs, rows, rows), rows, rows, n-1) + \
                    self._block_power(self._matmul_base_block(rhs, cols, rows), rows, cols, n-1)
            else:
                return self._block_power(self._matmul_base_block(rhs, rows, cols), rows, rows, n-1) + \
                    self._block_power(self._matmul_base_block(rhs, cols, cols), rows, cols, n-1)

    def _size(self):
        return self._args[2]._size()

    def _transpose_nonbatch(self):
        return self
