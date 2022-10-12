#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
from src.kernels.squared_exp import SquaredExp


class GaussianProcess(nn.Module):
    def __init__(self):
        super(GaussianProcess, self).__init__()

        # Default kernel
        self.kernel_ = SquaredExp()

        # Default signal variance
        self.signal_ = nn.Parameter(torch.tensor(1.), requires_grad=True)

        # Default noise variance
        self.noise_ = nn.Parameter(torch.tensor(1e-8), requires_grad=True)

    def covariance(self):
        return self.signal**2 * self.kernel(self.samples, self.samples) + self.noise**2 * torch.eye(self.samples.size(0), self.samples.size(0)).to(self.samples.device)

    def update(self):
        self.alpha_ = torch.linalg.solve(self.covariance(), self.target)

    def forward(self, x):
        # .unsqueeze(1)
        return torch.mm(self.kernel(self.samples, x).t(), self.alpha_)

    def train(self, iter=1000, verbose=True):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        for i in range(iter):
            K = self.covariance()
            loss = 0.5*(torch.mm(self.target.t(), torch.linalg.solve(K,
                        self.target)) + torch.linalg.slogdet(K).logabsdet)
            loss.backward(retain_graph=True)
            if verbose and i % 100 == 0:
                print(f"Iteration: {i}, Loss: {loss.item():0.2f}")
            opt.step()
            opt.zero_grad()
        self.update()

    # Kernel
    @property
    def kernel(self):
        return self.kernel_

    @kernel.setter
    def kernel(self, value):
        self.kernel_ = value

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
