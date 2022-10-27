#!/usr/bin/env python
# encoding: utf-8

import math
import torch
import torch.nn as nn


class Laplacian(nn.Module):
    def __init__(self):
        super(Laplacian, self).__init__()

        # Laplacian heat kernel hyperparameter
        self.eps_ = nn.Parameter(
            torch.tensor(0.01), requires_grad=True)

        # Covariance function length hyperparameter
        self.k_ = nn.Parameter(torch.tensor(0.1), requires_grad=True)

        # Covariance function signal variance hyperparameter
        self.sigma_ = nn.Parameter(torch.tensor(1.), requires_grad=True)

        # Covariance function noise variance hyperparameter
        self.sigma_n_ = nn.Parameter(torch.tensor(1e-3), requires_grad=True)

    def forward(self, x):
        l = -1 / self.eps_
        xx = torch.einsum('ij,ij->i', x, x)
        L = -2 * torch.mm(x, x.T) + xx.unsqueeze(1) + xx.unsqueeze(0)
        L *= l
        D = L.sum(dim=1).pow(-1)
        L = D.unsqueeze(1)*L*D.unsqueeze(0)
        L = (torch.eye(L.shape[0]).to(x.device) -
             L.sum(dim=1).pow(-1).unsqueeze(1)*L)/(1/4*self.eps_)

        return L

    def covariance_matern(self, x, nu=5):
        return torch.linalg.matrix_power(2*nu/self.k_**2 + self.forward(x), nu)*self.sigma_**2 + torch.eye(x.shape[0]).to(x.device)*self.sigma_n_**2

    def covariance_exp(self, x):
        L = self.forward(x)
        return (torch.eye(x.shape[0]).to(x.device) - self.k_.pow(2)/2*L + torch.linalg.matrix_power(self.k_.pow(2)/2*L, 2) - torch.linalg.matrix_power(self.k_.pow(2)/2*L, 3)/math.factorial(3)
                + torch.linalg.matrix_power(self.k_.pow(2) / 2*L, 4)/math.factorial(4))*self.sigma_**2 + torch.eye(x.shape[0]).to(x.device)*self.sigma_n_**2

    def train(self, x, y, iter=1000, verbose=True):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        for i in range(iter):
            # K = self.covariance_exp(x)
            K = self.covariance_matern(x)
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
