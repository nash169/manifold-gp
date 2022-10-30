#!/usr/bin/env python
# encoding: utf-8

import math
from aiosignal import Signal
import torch
import torch.nn as nn


class Laplacian(nn.Module):
    def __init__(self, X):
        super(Laplacian, self).__init__()

        # Similarity matrix
        # rows = torch.arange(indices.shape[0]).repeat_interleave(
        #     indices.shape[1]).unsqueeze(0)
        # cols = indices.reshape(1, -1)
        # values = -distances.reshape(1, -1).squeeze()
        # self.S_ = torch.sparse_coo_tensor(
        #     torch.cat((rows, cols), dim=0), values, (indices.shape[0], indices.shape[0])).to_dense()
        xx = torch.einsum('ij,ij->i', X, X)
        self.S_ = 2 * torch.mm(X, X.T) - xx.unsqueeze(1) - xx.unsqueeze(0)

        # Laplacian heat kernel hyperparameter
        self.eps_ = nn.Parameter(
            torch.tensor(0.1), requires_grad=True)

        # Covariance function length hyperparameter
        self.k_ = nn.Parameter(torch.tensor(1.), requires_grad=True)

        # Covariance function signal variance hyperparameter
        self.sigma_ = nn.Parameter(torch.tensor(1.), requires_grad=True)

        # Covariance function noise variance hyperparameter
        self.sigma_n_ = nn.Parameter(torch.tensor(1e-3), requires_grad=False)

    def forward(self):
        L = torch.exp(self.S_/self.eps_)
        D = L.sum(dim=1).pow(-1)
        L = D.unsqueeze(1)*L*D.unsqueeze(0)
        L = (torch.eye(L.shape[0]).to(self.S_.device) -
             L.sum(dim=1).pow(-1).unsqueeze(1)*L)/(1/4*self.eps_)

        return L

    def covariance_matern(self, nu=5):
        return torch.linalg.matrix_power(2*nu/self.k_**2*torch.eye(self.S_.shape[0]).to(self.S_.device) + self.forward(), -nu)*self.sigma_**2 + torch.eye(self.S_.shape[0]).to(self.S_.device)*self.sigma_n_**2

    # def covariance_exp(self, x):
    #     L = self.forward(x)
    #     return (torch.eye(x.shape[0]).to(x.device) - self.k_.pow(2)/2*L + torch.linalg.matrix_power(self.k_.pow(2)/2*L, 2) - torch.linalg.matrix_power(self.k_.pow(2)/2*L, 3)/math.factorial(3)
    #             + torch.linalg.matrix_power(self.k_.pow(2) / 2*L, 4)/math.factorial(4))*self.sigma_**2 + torch.eye(x.shape[0]).to(x.device)*self.sigma_n_**2

    def train(self, y, iter=1000, verbose=True):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        for i in range(iter):
            # K = self.covariance_exp(x)
            K = self.covariance_matern(5)
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
