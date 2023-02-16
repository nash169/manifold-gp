#!/usr/bin/env python
# encoding: utf-8


from time import time

from abc import abstractmethod
from typing import Optional

import torch
import numpy as np

import faiss
import faiss.contrib.torch_utils

import gpytorch
from gpytorch.constraints import Positive

from torch_geometric.utils import coalesce, scatter

from typing import Optional, Union, Callable, Tuple
from linear_operator import LinearOperator, settings, utils


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
        # register the raw parameter
        self.register_parameter(
            name='raw_epsilon', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1).to(nodes.device), requires_grad=False)
        )
        self.register_constraint("raw_epsilon", Positive())

        # KNN Faiss
        if self.nodes.is_cuda:
            res = faiss.StandardGpuResources()
            # config = faiss.GpuIndexFlatConfig()
            # config.useFloat16 = True
            self.knn = faiss.GpuIndexFlatL2(res, self.nodes.shape[1])
        else:
            self.knn = faiss.IndexFlatL2(self.nodes.shape[1])

        # Build graph
        self.generate_graph()

    @abstractmethod
    def spectral_density(self):
        raise NotImplementedError()

    # def forward(self, x1, x2, diag=False, **params):
    #     # covar = torch.mm(self.eigenfunctions(
    #     #     x1).T, self.spectral_density().T*self.eigenfunctions(x2))

    #     if diag:
    #         if torch.equal(x1, x2):
    #             return torch.sum(self.spectral_density().T*self.eigenfunctions(x1).pow(2), dim=0)
    #         else:
    #             return torch.sum(self.spectral_density().T*self.eigenfunctions(x1)*self.eigenfunctions(x2), dim=0)
    #         # return covar.diag()  # temporary solution (make diagonal calculation more efficient)
    #     else:
    #         return torch.mm(self.eigenfunctions(x1).T, self.spectral_density().T*self.eigenfunctions(x2))
    #         # return covar

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

    def laplacian(self, operator=False):
        # Diffusion Maps Normalization
        val = self.values.div(-2*self.epsilon.square()).exp().squeeze()
        deg = torch.zeros(self.nodes.shape[0]).to(self.nodes.device).scatter_add_(
            0, self.indices[0, :], val).scatter_add_(0, self.indices[1, :], val)
        val = val.div(deg[self.indices[0, :]]*deg[self.indices[1, :]])

        # Symmetric Laplacian Normalization
        deg = torch.zeros(self.nodes.shape[0]).to(self.nodes.device).scatter_add_(
            0, self.indices[0, :], val).scatter_add(0, self.indices[1, :], val).sqrt()
        val.div_(deg[self.indices[0, :]]*deg[self.indices[1, :]])

        if operator:
            return LaplacianOperator(val, self)
        else:
            return val

    def solve_laplacian(self):
        # Lanczos
        t0 = time()
        # with gpytorch.settings.max_root_decomposition_size(2*self.modes):
        #     self.eigenvalues, self.eigenvectors = self.laplacian(
        #         operator=True).diagonalization(method="lanczos")
        #     self.eigenvalues[0] = 0
        #     self.eigenvalues = self.eigenvalues[:self.modes]
        #     self.eigenvectors = self.eigenvectors.to_dense()[:, :self.modes]
        #     # self.eigenvectors = self.eigenvectors.to_dense()
        # self.eigenvalues, self.eigenvectors = LaplacianSymmetric(self.values, self).diagonalization()
        self.eigenvalues, self.eigenvectors = LaplacianRandomWalk(self.values, self).diagonalization()
        t1 = time()
        print("Model 1: %.4g sec" % (t1 - t0))

        # # Torch Sparse
        # t0 = time()
        # val = -self.laplacian()
        # val = torch.cat(
        #     (val.repeat(2), torch.ones(self.nodes.shape[0]).to(self.nodes.device)), dim=0)
        # idx = torch.cat((self.indices, torch.stack((self.indices[1, :], self.indices[0, :]), dim=0), torch.arange(
        #     self.nodes.shape[0]).repeat(2, 1).to(self.nodes.device)), dim=1)
        # L = torch.sparse_coo_tensor(
        #     idx, val, (self.nodes.shape[0], self.nodes.shape[0]))
        # self.eigenvalues, self.eigenvectors = torch.lobpcg(
        #     L, k=self.modes, largest=False)
        # t1 = time()
        # print("Model 1: %.4g sec" % (t1 - t0))

        # # Scipy Sparse
        # t0 = time()
        # from scipy.sparse import coo_matrix
        # from scipy.sparse.linalg import eigs, eigsh
        # val = -self.laplacian()
        # val = torch.cat(
        #     (val.repeat(2), torch.ones(self.nodes.shape[0]).to(self.nodes.device)), dim=0)
        # idx = torch.cat((self.indices, torch.stack((self.indices[1, :], self.indices[0, :]), dim=0), torch.arange(
        #     self.nodes.shape[0]).repeat(2, 1).to(self.nodes.device)), dim=1)
        # L = coo_matrix(
        #     (val.detach().cpu().numpy(), (idx[0, :].cpu().numpy(), idx[1, :].cpu().numpy())), shape=(self.nodes.shape[0], self.nodes.shape[0]))
        # T, V = eigsh(L, k=self.modes, which='SM')
        # self.eigenvalues = torch.from_numpy(T).float().to(self.nodes.device)
        # self.eigenvectors = torch.from_numpy(V).float().to(self.nodes.device)
        # t1 = time()
        # print("Model 1: %.4g sec" % (t1 - t0))

    def eigenfunctions(self, x):
        distances, indices = self.knn.search(
            x, self.neighbors)  # self.neighbors or fix it
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


class LaplacianOperator(LinearOperator):
    def __init__(self, laplacian, kernel):
        super(LaplacianOperator, self).__init__(
            laplacian, kernel=kernel)

    def _matmul(self, x):
        r = self._kwargs['kernel'].indices[0, :]
        c = self._kwargs['kernel'].indices[1, :]
        v = self._args[0]

        return x.index_add(0, r, v.view(-1, 1) * x[c], alpha=-1).index_add(0, c, v.view(-1, 1)*x[r], alpha=-1)

    def _size(self):
        dim = self._kwargs['kernel'].nodes.shape[0]
        return torch.Size([dim, dim])

    def _transpose_nonbatch(self):
        return self

    def evaluate_kernel(self):
        return self

    def matmul(self, other: Union[torch.Tensor, "LinearOperator"]) -> Union[torch.Tensor, "LinearOperator"]:
        return self._matmul(other)


class LaplacianSymmetric(LinearOperator):
    def __init__(self, values, kernel):
        super(LaplacianSymmetric, self).__init__(values, kernel=kernel)

        self.values = values.div(-2*kernel.epsilon.square()).exp().squeeze()
        self.indices = self._kwargs['kernel'].indices

        # Diffusion Maps Normalization
        deg = torch.zeros(self._kwargs['kernel'].nodes.shape[0]).to(self._kwargs['kernel'].nodes.device).scatter_add_(
            0, self.indices[0, :], self.values).scatter_add_(0, self.indices[1, :], self.values)
        self.values = self.values.div(deg[self.indices[0, :]]*deg[self.indices[1, :]])

        # Symmetric Laplacian Normalization
        deg = torch.zeros(self._kwargs['kernel'].nodes.shape[0]).to(self._kwargs['kernel'].nodes.device).scatter_add_(
            0, self.indices[0, :], self.values).scatter_add(0, self.indices[1, :], self.values).sqrt()
        self.values.div_(deg[self.indices[0, :]]*deg[self.indices[1, :]])

    def _matmul(self, x):
        # .div(self._kwargs['kernel'].epsilon.square().squeeze())
        return x.index_add(0, self.indices[0, :], self.values.view(-1, 1) * x[self.indices[1, :]], alpha=-1).index_add(0, self.indices[1, :], self.values.view(-1, 1)*x[self.indices[0, :]], alpha=-1)

    def _size(self):
        dim = self._kwargs['kernel'].nodes.shape[0]
        return torch.Size([dim, dim])

    def _transpose_nonbatch(self):
        return self

    def evaluate_kernel(self):
        return self

    def matmul(self, other: Union[torch.Tensor, "LinearOperator"]) -> Union[torch.Tensor, "LinearOperator"]:
        return self._matmul(other)

    def diagonalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with gpytorch.settings.max_root_decomposition_size(3*self._kwargs['kernel'].modes):
            evals, evecs = super().diagonalization(method="lanczos")
            # evecs = evecs.to_dense()
            # self.eigenvalues[0] = 0
            evals = evals[:self._kwargs['kernel'].modes]
            evecs = evecs.to_dense()[:, :self._kwargs['kernel'].modes]

        return evals, evecs


class LaplacianRandomWalk(LinearOperator):
    def __init__(self, values, kernel):
        super(LaplacianRandomWalk, self).__init__(values, kernel=kernel)

        self.values = values.div(-2*kernel.epsilon.square()).exp().squeeze().repeat(2)
        self.indices = torch.cat((self._kwargs['kernel'].indices, torch.stack(
            (self._kwargs['kernel'].indices[1, :], self._kwargs['kernel'].indices[0, :]), dim=0)), dim=1)

        # Diffusion Maps Normalization
        deg = torch.zeros(self._kwargs['kernel'].nodes.shape[0]).to(
            self._kwargs['kernel'].nodes.device).scatter_add_(0, self.indices[0, :], self.values)
        self.values = self.values.div(deg[self.indices[0, :]]*deg[self.indices[1, :]])

        # Random Walk Normalization
        self.degree = torch.zeros(self._kwargs['kernel'].nodes.shape[0]).to(
            self._kwargs['kernel'].nodes.device).scatter_add_(0, self.indices[0, :], self.values)
        self.values.div_(self.degree[self.indices[0, :]])

    def _matmul(self, x):
        return x.index_add(0, self.indices[0, :], self.values.view(-1, 1) * x[self.indices[1, :]], alpha=-1)

    def _size(self):
        dim = self._kwargs['kernel'].nodes.shape[0]
        return torch.Size([dim, dim])

    def _transpose_nonbatch(self):
        return self

    def evaluate_kernel(self):
        return self

    def matmul(self, other: Union[torch.Tensor, "LinearOperator"]) -> Union[torch.Tensor, "LinearOperator"]:
        return self._matmul(other)

    def diagonalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        evals, evecs = LaplacianSymmetric(self._args[0], self._kwargs['kernel']).diagonalization()

        return evals, evecs.div_(self.degree.sqrt().view(-1, 1))
