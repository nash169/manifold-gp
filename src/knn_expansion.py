#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

import faiss
import faiss.contrib.torch_utils


class KnnExpansion(nn.Module):
    def __init__(self):
        super(KnnExpansion, self).__init__()

        # Default neighborhoods
        self.k_ = 10

    def forward(self, x):
        d, i = self.knn.search(x, self.k)

        return torch.sum(self.alpha_[i].permute(2, 0, 1) * torch.exp(-0.5 * d/self.sigma**2), dim=2)

    # Weights
    @property
    def alpha(self):
        return self.alpha_

    @alpha.setter
    def alpha(self, value):
        self.alpha_ = value

    # KNN
    @property
    def knn(self):
        return self.knn_

    @knn.setter
    def knn(self, value):
        self.knn_ = value

    # Neighborhoods
    @property
    def k(self):
        return self.k_

    @k.setter
    def k(self, value):
        self.k_ = value

    # Sigma
    @property
    def sigma(self):
        return self.sigma_

    @sigma.setter
    def sigma(self, value):
        self.sigma_ = value
