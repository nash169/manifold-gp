#!/usr/bin/env python
# encoding: utf-8

from jaxtyping import Float
from torch import Tensor
from linear_operator import LinearOperator
from typing import Optional

from torch._C import Size


class ScaleWrapperOperator(LinearOperator):
    def __init__(self,
                 operator: LinearOperator,
                 scale: Tensor,
                 inverse_scale: Optional[bool] = False
                 ):
        super().__init__(
            operator,
            scale=scale,
            inverse_scale=inverse_scale
        )
        self.operator = operator
        self.scale = scale
        self.inverse_scale = inverse_scale

    def _matmul(self, rhs):
        return self.operator._matmul(rhs.contiguous())/self.scale if self.inverse_scale else self.operator._matmul(rhs.contiguous())*self.scale

    def _size(self) -> Size:
        return self.operator._size()

    def _transpose_nonbatch(self: LinearOperator) -> LinearOperator:
        return ScaleWrapperOperator(self.operator._transpose_nonbatch, self.scale, self.inverse_scale)
