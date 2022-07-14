#!/usr/bin/env python
# encoding: utf-8

import torch.nn as nn
from utils import squared_exp


class GaussianProcess(nn.Module):
    def __init__(self):
        super(GaussianProcess, self).__init__()

        lambda x, y: squared_exp(
            x, y, sigma=0.05, eta=10)

    def update(self):
        pass

    def forward(self, x):
        return x

    # Parameters
    @property
    def params(self):
        return self.params_

    @params.setter
    def params(self, value):
        self.params_ = nn.Parameter(value[0], requires_grad=value[1])

    # Kernel
    @property
    def kernel(self):
        return self.kernel_

    @kernel.setter
    def kernel(self, value):
        self.kernel_ = value

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
