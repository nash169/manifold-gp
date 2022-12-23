#!/usr/bin/env python
# encoding: utf-8


from abc import abstractmethod
from typing import Optional

import torch

import faiss
import faiss.contrib.torch_utils

import gpytorch
from gpytorch.constraints import Positive


class RiemannKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self, nodes: torch.Tensor, neighbors: Optional[int] = 2, modes: Optional[int] = 10, **kwargs):
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
            self.knn = faiss.GpuIndexFlatL2(res, self.nodes.shape[1])
        else:
            self.knn = faiss.IndexFlatL2(self.nodes.shape[1])

        # Build graph (for the moment we leave the generation of the graph here. It could be moved before the training only)
        self.generate_graph()

        # # Compute eigenvalues and eigenvectors (for the moment this happens in the evaluation of the model)
        # self.solve_laplacian()

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
        # self.knn.reset()
        self.knn.train(self.nodes)
        self.knn.add(self.nodes)
        dist, idx = self.knn.search(self.nodes, self.neighbors+1)
        self.values = dist[:, 1:]
        self.indices = idx

    def laplacian_matrix(self):
        # Similarity matrix
        val = self.values.div(-2*self.epsilon.square()).exp()

        # Degree matrix
        deg = val.sum(dim=1)

        # Symmetric normalization
        val = val.div(deg.sqrt().unsqueeze(1)).div(
            deg.sqrt()[self.indices[:, 1:]])

        # Laplacian matrix
        val = torch.cat(
            (torch.ones(self.nodes.shape[0], 1).to(self.nodes.device), -val), dim=1)

        rows = torch.arange(self.indices.shape[0]).repeat_interleave(
            self.indices.shape[1]).unsqueeze(0).to(self.nodes.device)
        cols = self.indices.reshape(1, -1)
        val = val.reshape(1, -1).squeeze()

        return torch.sparse_coo_tensor(torch.cat((rows, cols), dim=0), val, (self.indices.shape[0], self.indices.shape[0]))

    def solve_laplacian(self):
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import eigs
        L = self.laplacian_matrix()
        indices = L.coalesce().indices().cpu().detach().numpy()
        values = L.coalesce().values().cpu().detach().numpy()
        Ls = coo_matrix(
            (values, (indices[0, :], indices[1, :])), shape=L.shape)
        T, V = eigs(Ls, k=self.modes, which='SR')
        self.eigenvalues = torch.from_numpy(T).float().to(self.nodes.device)
        self.eigenvectors = torch.from_numpy(V).float().to(self.nodes.device)

        # self.eigenvalues, self.eigenvectors = torch.lobpcg(
        #     self.laplacian_matrix(), k=self.modes, largest=False)

    def eigenfunctions(self, x):
        distances, indices = self.knn.search(x, self.neighbors)

        return torch.sum(self.eigenvectors[indices].permute(2, 0, 1) * distances.div(-2*self.epsilon.square()).exp(), dim=2)
