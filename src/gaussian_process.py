#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
from src.utils import squared_exp


class GaussianProcess(nn.Module):
    def __init__(self):
        super(GaussianProcess, self).__init__()

        # Default kernel
        self.kernel_ = lambda x, y, *args: squared_exp(x, y, *args)
        self.params_ = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        # Default signal variance
        self.signal_ = nn.Parameter(torch.tensor(1.), requires_grad=True)

        # Default noise variance
        self.noise_ = nn.Parameter(torch.tensor(1e-8), requires_grad=True)

    def update(self):
        self.alpha_ = torch.linalg.solve(self.signal * self.kernel(self.samples, self.samples, *self.params_.tolist(
        )) + self.noise * torch.eye(self.samples.size(0), self.samples.size(0)).to(self.samples.device), self.target)

    def forward(self, x):
        return torch.mm(self.kernel(self.samples, x, *self.params_.tolist()).t(), self.alpha_)

    # Kernel
    @property
    def kernel(self):
        return self.kernel_

    @kernel.setter
    def kernel(self, value):
        self.kernel_ = lambda x, y, *args: value[0](x, y, *args)
        self.params_ = nn.Parameter(value[1], requires_grad=value[2])

    # Signal variance
    @property
    def signal(self):
        return self.signal_

    @signal.setter
    def signal(self, value):
        self.signal_ = nn.Parameter(value[0], requires_grad=value[1])

    # Noise variance
    @property
    def noise(self):
        return self.noise_

    @noise.setter
    def noise(self, value):
        self.noise_ = nn.Parameter(value[0], requires_grad=value[1])

    # Training point
    @property
    def samples(self):
        return self.samples_

    @samples.setter
    def samples(self, value):
        self.samples_ = value

    # Ground truth
    @property
    def target(self):
        return self.target_

    @target.setter
    def target(self, value):
        self.target_ = value
