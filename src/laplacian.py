#!/usr/bin/env python
# encoding: utf-8

import math
import torch
import torch.nn as nn
import numpy as np
from time import time


class Laplacian(nn.Module):
    def __init__(self):
        super(Laplacian, self).__init__()

        # Default neighborhoods
        self.eps_ = nn.Parameter(
            torch.tensor(0.1), requires_grad=True)
        self.sigma_ = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, x):
        l = -1. / self.eps_
        xx = torch.einsum('ij,ij->i', x, x)
        L = -2 * torch.mm(x, x.T) + xx.unsqueeze(1) + xx.unsqueeze(0)
        L *= l
        D = L.sum(dim=1).pow(-1)
        L = D.unsqueeze(1)*L*D.unsqueeze(0)
        L = torch.eye(L.shape[0]).to(x.device) - \
            L.sum(dim=1).pow(-1).unsqueeze(1)*L

        return L

    def power(self, L):
        return torch.eye(L.shape[0]).to(L.device) - self.sigma_.pow(2)/2*L + torch.linalg.matrix_power(self.sigma_.pow(2)/2*L, 2) - torch.linalg.matrix_power(self.sigma_.pow(2)/2*L, 3)/math.factorial(3) + torch.linalg.matrix_power(self.sigma_.pow(2)/2*L, 4)/math.factorial(4)

    def train(self, y, iter=1000, verbose=True):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        for i in range(iter):
            K = self.power(self.forward(self.samples))
            loss = 0.5 * \
                (torch.mm(y.t(), torch.linalg.solve(K, y)) +
                 torch.linalg.slogdet(K).logabsdet)
            # t0 = time()
            loss.backward(retain_graph=True)
            # t1 = time()
            # print("Time: %.2g sec" % (t1 - t0))
            if verbose and i % 100 == 0:
                print(f"Iteration: {i}, Loss: {loss.item():0.2f}")
            opt.step()
            opt.zero_grad()

    # Samples
    @property
    def samples(self):
        return self.samples_

    @samples.setter
    def samples(self, value):
        self.samples_ = value

    # # KNN
    # @property
    # def knn(self):
    #     return self.knn_

    # @knn.setter
    # def knn(self, value):
    #     self.knn_ = value

    # # KNN
    # @property
    # def k(self):
    #     return self.k_

    # @k.setter
    # def k(self, value):
    #     self.k_ = value
