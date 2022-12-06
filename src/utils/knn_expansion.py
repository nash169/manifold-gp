#!/usr/bin/env python
# encoding: utf-8

from typing import Optional

import torch
import torch.nn as nn

import faiss
import faiss.contrib.torch_utils

from src.utils.bases import radial_basis


class KnnExpansion(nn.Module):
    def __init__(self, knn, neighbors: Optional[int] = 2):
        super(KnnExpansion, self).__init__()

        self.knn
        self.neighbors = neighbors

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
