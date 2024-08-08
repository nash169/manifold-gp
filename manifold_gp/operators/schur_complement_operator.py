#!/usr/bin/env python
# encoding: utf-8

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from linear_operator import LinearOperator
from linear_operator.operators import MaskedLinearOperator


class SchurComplementOperator(LinearOperator):
    def __init__(
        self,
        base: Float[LinearOperator, "*batch M0 M0"],
        mask: Bool[Tensor, "M0"],
    ):
        super().__init__(
            base,
            mask
        )

        self.base = base
        self.mask = mask

    def _matmul(self, rhs):
        tmp = MaskedLinearOperator(self.base, torch.ones(self.base.shape[0], dtype=torch.bool), self.mask)._matmul(rhs.contiguous())
        out = MaskedLinearOperator(self.base, ~self.mask, ~self.mask).solve(tmp[~self.mask])
        out = MaskedLinearOperator(self.base, self.mask, ~self.mask)._matmul(out)
        return tmp[self.mask] - out

    def _size(self):
        return torch.Size([self.mask.sum(), self.mask.sum()])

    def _transpose_nonbatch(self):
        return SchurComplementOperator(self.base._transpose_nonbatch, self.mask)
