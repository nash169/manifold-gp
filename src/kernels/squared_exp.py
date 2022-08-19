#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn


class SquaredExp(nn.Module):
    def __init__(self, l=1.):
        super(SquaredExp, self).__init__()

        self.sigma_ = nn.Parameter(torch.tensor(l), requires_grad=True)

    def forward(self, x, y):
        l = -.5 / self.sigma**2
        xx = torch.einsum('ij,ij->i', x, x).unsqueeze(1)
        yy = torch.einsum('ij,ij->i', y, y).unsqueeze(0)
        k = -2 * torch.mm(x, y.T) + xx + yy
        k *= l
        return torch.exp(k)

    # Sigma variance
    @property
    def sigma(self):
        return self.sigma_

    @sigma.setter
    def sigma(self, value):
        self.sigma_ = nn.Parameter(value[0], requires_grad=value[1])
