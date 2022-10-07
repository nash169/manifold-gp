#!/usr/bin/env python
# encoding: utf-8

from turtle import shape
import torch
import torch.nn as nn

from src.utils import squared_exp
from src.knn_expansion import KnnExpansion


class RiemannExp(nn.Module):
    def __init__(self, l=1.):
        super(RiemannExp, self).__init__()

        self.sigma_ = nn.Parameter(torch.tensor(l), requires_grad=True)

    def spectral(self):
        s = (-0.5*self.sigma.pow(2)*self.eigenvalues).exp()
        return s/s.sum()

    def forward(self, x, y):
        return torch.mm(self.eigenfunctions(x).T, self.spectral().unsqueeze(1)*self.eigenfunctions(y))

    # Sigma variance
    @property
    def sigma(self):
        return self.sigma_

    @sigma.setter
    def sigma(self, value):
        self.sigma_ = nn.Parameter(value[0], requires_grad=value[1])

    # Eigenvalues
    @property
    def eigenvalues(self):
        return self.eigenvalues_

    @eigenvalues.setter
    def eigenvalues(self, value):
        self.eigenvalues_ = value

    # Eigenfunction
    @property
    def eigenfunctions(self):
        return self.eigenfunctions_

    @eigenfunctions.setter
    def eigenfunctions(self, value):
        self.eigenfunctions_ = value
