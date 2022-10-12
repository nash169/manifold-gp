#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn


class RiemannMatern(nn.Module):
    def __init__(self, l=1.):
        super(RiemannMatern, self).__init__()

        self.k_ = nn.Parameter(torch.tensor(l), requires_grad=True)

        self.ni_ = 3

        self.d_ = 1.0

    def spectral(self):
        s = (2*self.ni_ / self.k**2 +
             self.eigenvalues).pow(-self.ni_ - self.d_/2)
        return s/s.sum()

    def forward(self, x, y):
        return torch.mm(self.eigenfunctions(x).T, self.spectral().unsqueeze(1)*self.eigenfunctions(y))

    # Sigma variance
    @property
    def k(self):
        return self.k_

    @k.setter
    def k(self, value):
        self.k_ = nn.Parameter(value[0], requires_grad=value[1])

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
