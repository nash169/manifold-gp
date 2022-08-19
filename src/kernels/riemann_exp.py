#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from src.utils import squared_exp


class RiemannExp(nn.Module):
    def __init__(self, l=1.):
        super(RiemannExp, self).__init__()

        self.sigma_ = nn.Parameter(torch.tensor(l), requires_grad=True)
        self.kernel_ = lambda x, y: squared_exp(x, y, 0.1)

    def spectral(self):
        s = (-0.5*self.sigma.pow(2)*self.eigenvalues).exp()
        return s/s.sum()

    def forward(self, x, y):
        return torch.mm(torch.mm(self.kernel_(x, self.samples), self.eigenvectors).sum(dim=1).unsqueeze(1), torch.mm(self.spectral().unsqueeze(0), torch.mm(self.eigenvectors.t(), self.kernel_(self.samples, y))))

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

    # Eigenvectors
    @property
    def eigenvectors(self):
        return self.eigenvectors_

    @eigenvectors.setter
    def eigenvectors(self, value):
        self.eigenvectors_ = value

    # Eigenvectors
    @property
    def samples(self):
        return self.samples_

    @samples.setter
    def samples(self, value):
        self.samples_ = value
