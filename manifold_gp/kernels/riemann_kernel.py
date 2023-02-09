#!/usr/bin/env python
# encoding: utf-8

from time import time

from abc import abstractmethod
from typing import Optional

import torch

import faiss
import faiss.contrib.torch_utils

import gpytorch
from gpytorch.constraints import Positive


class RiemannKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self, nodes: torch.Tensor, neighbors: Optional[int] = 2, modes: Optional[int] = 10, alpha: Optional[float] = 1.0, laplacian: Optional[str] = "normalized", **kwargs):
        super(RiemannKernel, self).__init__(**kwargs)

        # Hyperparameter
        self.nodes = nodes.unsqueeze(-1) if nodes.ndimension() == 1 else nodes
        self.neighbors = neighbors
        self.modes = modes

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

        # Build graph (for the moment we leave the generation of the graph here. It could be moved before the training only)
        self.generate_graph()

        # Store Laplacian parameters
        self.alpha = alpha
        self.ltype = laplacian
        # The eigen decomposition of the Laplacian is not performed here but when the model is evaluated. If you want to evaluate
        # the kernel call solve_laplacian method.

    @abstractmethod
    def spectral_density(self):
        raise NotImplementedError()

    def forward(self, x1, x2, diag=False, **params):
        covar = torch.mm(self.eigenfunctions(
            x1).T, self.spectral_density().T*self.eigenfunctions(x2))

        if diag:
            return covar.diag()  # temporary solution (make diagonal calculation more efficient)
        else:
            return covar

    @property
    def epsilon(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_epsilon_constraint.transform(self.raw_epsilon)

    @epsilon.setter
    def epsilon(self, value):
        return self._set_epsilon(value)

    def _set_epsilon(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_epsilon)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(
            raw_epsilon=self.raw_epsilon_constraint.inverse_transform(value))

    def generate_graph(self):
        self.knn.reset()
        self.knn.train(self.nodes)
        self.knn.add(self.nodes)
        dist, idx = self.knn.search(self.nodes, self.neighbors+1)
        self.values = dist[:, 1:]
        self.indices = idx

    # def laplacian_matrix(self):
    #     # Similarity matrix
    #     val = self.values.div(-2*self.epsilon.square()).exp()

    #     # Degree matrix
    #     deg = val.sum(dim=1)

    #     # Symmetric normalization
    #     val = val.div(deg.sqrt().unsqueeze(1)).div(
    #         deg.sqrt()[self.indices[:, 1:]])

    #     # Laplacian matrix
    #     val = torch.cat(
    #         (torch.ones(self.nodes.shape[0], 1).to(self.nodes.device), -val), dim=1)

    #     rows = torch.arange(self.indices.shape[0]).repeat_interleave(
    #         self.indices.shape[1]).unsqueeze(0).to(self.nodes.device)
    #     cols = self.indices.reshape(1, -1)
    #     val = val.reshape(1, -1).squeeze()

    #     return torch.sparse_coo_tensor(torch.cat((rows, cols), dim=0), val, (self.indices.shape[0], self.indices.shape[0]))

    # It follows: https://www.jmlr.org/papers/volume8/hein07a/hein07a.pdf

    def laplacian(self, alpha=1, type="normalized"):
        # Re-normalized kernel from Diffusion
        val = self.values.div(-2*self.epsilon.square()).exp()
        deg = val.sum(dim=1).pow(alpha)
        val = val.div(deg.unsqueeze(1)).div(deg[self.indices[:, 1:]])

        # Select the normalization type
        if type == "randomwalk":
            val = torch.cat((torch.ones(self.nodes.shape[0], 1).to(
                self.nodes.device), -val.div(val.sum(dim=1).unsqueeze(1))), dim=1)
        elif type == "unnormalized":
            val = torch.cat((torch.ones(self.nodes.shape[0], 1).to(
                self.nodes.device), -val), dim=1)
        elif type == "normalized":
            deg = val.sum(dim=1).sqrt()
            val = torch.cat((torch.ones(self.nodes.shape[0], 1).to(
                self.nodes.device), -val.div(deg.unsqueeze(1)).div(deg[self.indices[:, 1:]])), dim=1)

        return val

    def to_sparse(self, val):
        rows = torch.arange(self.indices.shape[0]).repeat_interleave(
            self.indices.shape[1]).unsqueeze(0).to(self.nodes.device)
        cols = self.indices.reshape(1, -1)
        val = val.reshape(1, -1).squeeze()

        return torch.sparse_coo_tensor(torch.cat((rows, cols), dim=0), val, (self.indices.shape[0], self.indices.shape[0]))

    def solve_laplacian(self):
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import eigs
        # L = self.laplacian_matrix()
        L = self.to_sparse(self.laplacian(alpha=1, type="normalized"))
        indices = L.coalesce().indices().cpu().detach().numpy()
        values = L.coalesce().values().cpu().detach().numpy()
        Ls = coo_matrix(
            (values, (indices[0, :], indices[1, :])), shape=L.shape)
        T, V = eigs(Ls, k=self.modes, which='SR')
        self.eigenvalues = torch.from_numpy(T).float().to(self.nodes.device)
        # self.eigenvectors = np.sqrt(
        #     self.nodes.shape[0])*torch.from_numpy(V).float().to(self.nodes.device).div(deg.unsqueeze(-1))
        self.eigenvectors = torch.from_numpy(V).float().to(self.nodes.device)

        # self.eigenvalues, self.eigenvectors = torch.lobpcg(
        #     self.laplacian_matrix(), k=self.modes, largest=False)

        # self.eigenvalues, self.eigenvectors = torch.lobpcg(self.to_sparse(
        #     self.laplacian(alpha=self.alpha, type=self.ltype)), k=self.modes, largest=False, tol=1e-4, niter=-1)

    def eigenfunctions(self, x):
        distances, indices = self.knn.search(
            x, self.neighbors)  # self.neighbors or fix it
        # return torch.sum(self.eigenvectors[indices].permute(2, 0, 1) * distances.div(-2*self.epsilon.square()).exp(), dim=2)
        return self.inverse_distance_weighting(distances, self.eigenvectors[indices].permute(2, 0, 1))

    def inverse_distance_weighting(self, d, y):
        u = torch.zeros(y.shape[0], y.shape[1]).to(d.device)
        idx_zero = torch.any(d <= 1e-8, 1)
        u[:, idx_zero] = y[:, idx_zero, 0]
        u[:, ~idx_zero] = torch.sum(
            y[:, ~idx_zero, :] / d[~idx_zero, :], dim=2).div(torch.sum(d[~idx_zero, :].pow(-1), dim=1))
        return u
