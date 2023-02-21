#!/usr/bin/env python
# encoding: utf-8

from abc import abstractmethod
from typing import Optional

import torch
import faiss
import faiss.contrib.torch_utils

import gpytorch
from gpytorch.constraints import Positive

from torch_geometric.utils import coalesce

from typing import Optional, Tuple, Union
from linear_operator import LinearOperator


class RiemannKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self, nodes: torch.Tensor, neighbors: Optional[int] = 2, modes: Optional[int] = 10, support_kernel: Optional[gpytorch.kernels.Kernel] = gpytorch.kernels.RBFKernel(), **kwargs):
        super(RiemannKernel, self).__init__(**kwargs)

        # Hyperparameter
        self.nodes = nodes.unsqueeze(-1) if nodes.ndimension() == 1 else nodes
        self.neighbors = neighbors
        self.modes = modes
        self.support_kernel = support_kernel

        # Heat kernel length parameter for Laplacian approximation
        self.register_parameter(
            name='raw_epsilon', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1).to(nodes.device), requires_grad=True)
        )
        self.register_constraint("raw_epsilon", Positive())

        # KNN Faiss
        if self.nodes.is_cuda:
            res = faiss.StandardGpuResources()
            self.knn = faiss.GpuIndexFlatL2(res, self.nodes.shape[1])
        else:
            self.knn = faiss.IndexFlatL2(self.nodes.shape[1])

        # Build graph
        self.generate_graph()

    @abstractmethod
    def spectral_density(self):
        raise NotImplementedError()

    def base(self, x1, x2, diag=False):
        if diag:
            if torch.equal(x1, x2):
                return torch.sum(self.spectral_density().T*self.eigenfunctions(x1).pow(2), dim=0)
            else:
                return torch.sum(self.spectral_density().T*self.eigenfunctions(x1)*self.eigenfunctions(x2), dim=0)
        else:
            return torch.mm(self.eigenfunctions(x1).T, self.spectral_density().T*self.eigenfunctions(x2))

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            if torch.equal(x1, x2):
                return self.variance(x1)
            else:
                return self.base(x1, x2, diag=True) + self.support_kernel.forward(x1, x2, diag=True)*torch.sqrt(self.variance(x1) - self.base(x1, x1, diag=True))*torch.sqrt(self.variance(x2) - self.base(x2, x2, diag=True))
        else:
            return self.base(x1, x2) + self.support_kernel.forward(x1, x2)*torch.outer(torch.sqrt(self.variance(x1) - self.base(x1, x1, diag=True)), torch.sqrt(self.variance(x2) - self.base(x2, x2, diag=True)))

    @property
    def epsilon(self):
        return self.raw_epsilon_constraint.transform(self.raw_epsilon)

    @epsilon.setter
    def epsilon(self, value):
        return self._set_epsilon(value)

    def _set_epsilon(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_epsilon)
        self.initialize(
            raw_epsilon=self.raw_epsilon_constraint.inverse_transform(value))

    def generate_graph(self):
        # KNN Search
        self.knn.reset()
        self.knn.train(self.nodes)
        self.knn.add(self.nodes)
        val, idx = self.knn.search(self.nodes, self.neighbors+1)

        # Symmetric KNN
        rows = torch.arange(idx.shape[0]).repeat_interleave(
            idx.shape[1]-1).to(self.nodes.device)
        cols = idx[:, 1:].reshape(1, -1).squeeze()
        val = val[:, 1:].reshape(1, -1).squeeze()
        split = cols > rows
        rows, cols = torch.cat([rows[split], cols[~split]], dim=0), torch.cat(
            [cols[split], rows[~split]], dim=0)
        idx = torch.stack([rows, cols], dim=0)
        val = torch.cat([val[split], val[~split]])
        self.indices, self.values = coalesce(idx, val, reduce='mean')

    def laplacian(self, operator):
        if operator == 'symmetric':
            # Adjacency Matrix
            idx = self.indices
            val = self.values.div(-2*self.epsilon.square()).exp().squeeze()

            # Diffusion Maps Normalization
            deg = torch.zeros(self.nodes.shape[0]).to(self.nodes.device).scatter_add_(0, idx[0, :], val).scatter_add_(0, idx[1, :], val)
            val = val.div(deg[idx[0, :]]*deg[idx[1, :]])

            # Symmetric Laplacian Normalization
            deg = torch.zeros(self.nodes.shape[0]).to(self.nodes.device).scatter_add_(0, idx[0, :], val).scatter_add(0, idx[1, :], val).sqrt()
            val.div_(deg[idx[0, :]]*deg[idx[1, :]])

            return LaplacianSymmetric(val, idx, self)
        elif operator == 'randomwalk':
            # Adjacency Matrix
            val = self.values.div(-2*self.epsilon.square()).exp().squeeze().repeat(2)
            idx = torch.cat((self.indices, torch.stack((self.indices[1, :], self.indices[0, :]), dim=0)), dim=1)

            # Diffusion Maps Normalization
            deg = torch.zeros(self.nodes.shape[0]).to(self.nodes.device).scatter_add_(0, idx[0, :], val)
            val = val.div(deg[idx[0, :]]*deg[idx[1, :]])

            # Random Walk Normalization
            deg = torch.zeros(self.nodes.shape[0]).to(self.nodes.device).scatter_add_(0, idx[0, :], val)
            val.div_(deg[idx[0, :]])

            return LaplacianRandomWalk(val, deg.pow(-1), idx, self)

    def generate_eigenpairs(self):
        self.eigenvalues, self.eigenvectors = self.laplacian(operator='randomwalk').diagonalization()

    def eigenfunctions(self, x):
        distances, indices = self.knn.search(x, self.neighbors)  # self.neighbors or fix it
        # return torch.sum(self.eigenvectors[indices].permute(2, 0, 1) * distances.div(-2*self.epsilon.square()).exp(), dim=2)
        return self.inverse_distance_weighting(distances, self.eigenvectors[indices].permute(2, 0, 1))

    def variance(self, x):
        distances, indices = self.knn.search(x, self.neighbors)

        return torch.sum(self.spectral_density().T*self.inverse_distance_weighting(distances, self.eigenvectors[indices].pow(2).permute(2, 0, 1)), dim=0)

    def inverse_distance_weighting(self, d, y):
        u = torch.zeros(y.shape[0], y.shape[1]).to(d.device)
        idx_zero = torch.any(d <= 1e-8, 1)
        u[:, idx_zero] = y[:, idx_zero, 0]
        u[:, ~idx_zero] = torch.sum(
            y[:, ~idx_zero, :] / d[~idx_zero, :], dim=2).div(torch.sum(d[~idx_zero, :].pow(-1), dim=1))
        return u


class LaplacianSymmetric(LinearOperator):
    def __init__(self, values, indices, kernel):
        super(LaplacianSymmetric, self).__init__(values, indices, kernel=kernel)

    def _matmul(self, x):
        return x.index_add(0, self._args[1][0, :], self._args[0].view(-1, 1) * x[self._args[1][1, :]], alpha=-1) \
                .index_add(0, self._args[1][1, :], self._args[0].view(-1, 1)*x[self._args[1][0, :]], alpha=-1)

    def _size(self):
        return torch.Size([self._kwargs['kernel'].nodes.shape[0], self._kwargs['kernel'].nodes.shape[0]])

    def _transpose_nonbatch(self):
        return self

    def diagonalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with gpytorch.settings.max_root_decomposition_size(3*self._kwargs['kernel'].modes):
            evals, evecs = super().diagonalization(method="lanczos")
            evals = evals[:self._kwargs['kernel'].modes]
            evecs = evecs.to_dense()[:, :self._kwargs['kernel'].modes]
        return evals, evecs


class LaplacianRandomWalk(LinearOperator):
    def __init__(self, values, degree, indices, kernel):
        super(LaplacianRandomWalk, self).__init__(values, degree, indices, kernel=kernel)

    def _matmul(self, x):
        return x.index_add(0, self._args[2][0, :], self._args[0].view(-1, 1) * x[self._args[2][1, :]], alpha=-1)

    def _size(self):
        return torch.Size([self._kwargs['kernel'].nodes.shape[0], self._kwargs['kernel'].nodes.shape[0]])

    def _transpose_nonbatch(self):
        return self

    def diagonalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        evals, evecs = self._kwargs['kernel'].laplacian(operator='symmetric').diagonalization()
        return evals, evecs.mul_(self._args[1].sqrt().view(-1, 1))
