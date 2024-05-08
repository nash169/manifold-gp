#!/usr/bin/env python
# encoding: utf-8

from torch import Tensor
from linear_operator.operators._linear_operator import LinearOperator
from torch._C import Size


class NoiseWrapperOperator(LinearOperator):
    def __init__(self,
                 operator: LinearOperator,
                 noise: Tensor,
                 ):
        super().__init__(
            operator,
            noise=noise
        )
        self.operator = operator
        self.noise = noise

    def _matmul(self, rhs):
        return self.operator._matmul(rhs.contiguous() - self.noise*self.operator._matmul(rhs.contiguous() - self.noise*self.operator._matmul(rhs.contiguous())))

    def _size(self) -> Size:
        return self.operator._size()

    def _transpose_nonbatch(self: LinearOperator) -> LinearOperator:
        return NoiseWrapperOperator(self.operator._transpose_nonbatch, self.noise)
