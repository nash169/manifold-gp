#!/usr/bin/env python
# encoding: utf-8

from abc import abstractmethod

import math
import torch
from torch import Tensor
import faiss
import faiss.contrib.torch_utils

import gpytorch
from gpytorch.constraints import Positive, GreaterThan

from torch_geometric.utils import coalesce
from torch_geometric.nn import radius

from typing import Optional, Tuple, Union
from linear_operator import LinearOperator
from linear_operator.operators import LowRankRootLinearOperator, MatmulLinearOperator, RootLinearOperator

from manifold_gp.priors.inverse_gamma_prior import InverseGammaPrior


class RiemannKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self,
                 nodes: torch.Tensor, neighbors: Optional[int] = 2,
                 operator: Optional[str] = "randomwalk",
                 modes: Optional[int] = 10,
                 ball_scale: Optional[float] = 1.0,
                 prior_bandwidth: Optional[bool] = False,
                 **kwargs):
        super(RiemannKernel, self).__init__(**kwargs)

        # Hyperparameter
        self.nodes = nodes.unsqueeze(-1) if nodes.ndimension() == 1 else nodes
        self.neighbors = neighbors
        self.operator = operator
        self.modes = modes
        self.ball_scale = ball_scale
        self.prior_bandwidth = prior_bandwidth

        # KNN Faiss
        if self.nodes.is_cuda:
            res = faiss.StandardGpuResources()
            self.knn = faiss.GpuIndexIVFFlat(res, self.nodes.shape[1], 1, faiss.METRIC_L2)
        else:
            self.knn = faiss.IndexFlatL2(self.nodes.shape[1])

        # Build graph
        self.generate_graph()

        # Heat kernel length parameter for Laplacian approximation
        self.register_parameter(
            name='raw_epsilon', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1).to(nodes.device), requires_grad=True)
        )

        if prior_bandwidth:
            self.register_prior(
                "epsilon_prior", gpytorch.priors.GammaPrior(self.eps_concentration, self.eps_rate), self._epsilon_param, self._epsilon_closure
            )

        # self.register_constraint("raw_epsilon", GreaterThan(self.eps_gte))
        self.register_constraint("raw_epsilon", Positive())

    @abstractmethod
    def spectral_density(self):
        raise NotImplementedError()

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **kwargs) -> Tensor:
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)
        z1 = self.features(x1, normalize=True, c=self.ball_scale)
        if not x1_eq_x2:
            z2 = self.features(x2, normalize=True, c=self.ball_scale)
        else:
            z2 = z1

        if diag:
            return (z1 * z2).sum(-1)
        if x1_eq_x2:
            # Exploit low rank structure, if there are fewer features than data points
            if z1.size(-1) < z2.size(-2):
                return LowRankRootLinearOperator(z1)
            else:
                return RootLinearOperator(z1)
        else:
            return MatmulLinearOperator(z1, z2.transpose(-1, -2))

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

    def _epsilon_param(self, m):
        return m.epsilon

    def _epsilon_closure(self, m, v):
        return m._set_epsilon(v)

    def generate_graph(self):
        # KNN Search
        self.knn.reset()
        self.knn.train(self.nodes)
        self.knn.add(self.nodes)
        val, idx = self.knn.search(self.nodes, self.neighbors+1)

        # Calculate epsilon lower bound (99% kernel decay)
        min_keval = torch.tensor(1e-3).to(self.nodes.device)
        self.eps_gte = val[:, 1].max().div(-4*min_keval.log()).sqrt()

        # Gamma distribution hyperparameters
        if self.prior_bandwidth:
            p_50 = val[:, 1:].sqrt().mean(dim=1).sort()[0][int(round(val.shape[0]*0.50))]
            k_std = 4*p_50/(p_50-self.eps_gte)**2
            self.eps_rate = k_std
            self.eps_concentration = k_std * p_50 + 1

        # Make KNN Symmetric
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
            val = self.values.div(-4*self.epsilon.square()).exp().squeeze()

            # Diffusion Maps Normalization
            deg = torch.zeros(self.nodes.shape[0]).to(self.nodes.device).scatter_add_(0, idx[0, :], val).scatter_add_(0, idx[1, :], val)
            val = val.div(deg[idx[0, :]]*deg[idx[1, :]])

            # Symmetric Laplacian Normalization
            deg = torch.zeros(self.nodes.shape[0]).to(self.nodes.device).scatter_add_(0, idx[0, :], val).scatter_add(0, idx[1, :], val).sqrt()
            val.div_(deg[idx[0, :]]*deg[idx[1, :]])

            return LaplacianSymmetric(val, idx, self.epsilon, self)
        elif operator == 'randomwalk':
            # Adjacency Matrix
            val = self.values.div(-4*self.epsilon.square()).exp().squeeze().repeat(2)
            idx = torch.cat((self.indices, torch.stack((self.indices[1, :], self.indices[0, :]), dim=0)), dim=1)

            # Diffusion Maps Normalization
            deg = torch.zeros(self.nodes.shape[0]).to(self.nodes.device).scatter_add_(0, idx[0, :], val)
            val = val.div(deg[idx[0, :]]*deg[idx[1, :]])

            # Random Walk Normalization
            deg = torch.zeros(self.nodes.shape[0]).to(self.nodes.device).scatter_add_(0, idx[0, :], val)
            val.div_(deg[idx[0, :]])

            return LaplacianRandomWalk(val, deg.pow(-1), idx, self.epsilon, self)

    def generate_eigenpairs(self):
        evals, evecs = self.laplacian(operator=self.operator).diagonalization()
        self.eigenvalues = evals.detach()
        self.eigenvectors = evecs.detach()

        # calculate normalization terms
        if self.operator == "randomwalk":
            # self.normalization = (self.spectral_density() * self.eigenvectors.square()).sum() / self.nodes.shape[0]
            self.normalization = (self.spectral_density().div((1 - self.epsilon.square() * self.eigenvalues).square()) * self.eigenvectors.square()).sum() / self.nodes.shape[0]
        elif self.operator == "symmetric":
            self.normalization = self.spectral_density().sum() / self.nodes.shape[0]
        else:
            print("Operator not implemented.")

    def features(self, x: Tensor, normalize: bool = False, c: float = 1.0) -> Tensor:
        # features weights
        weights = (self.spectral_density() / self.normalization).sqrt()

        if torch.equal(x, self.nodes):
            return weights * self.eigenvectors
        else:
            # Degree Matrix Train Set (this part does not have to be done online)
            idx = torch.cat((self.indices, torch.stack((self.indices[1, :], self.indices[0, :]), dim=0)), dim=1)
            deg = torch.zeros(self.nodes.shape[0]).to(self.nodes.device).scatter_add_(0, idx[0, :], self.values.div(-4*self.epsilon.square()).exp().squeeze().repeat(2))

            # KNN Search
            val, idx_t = self.knn.search(x, self.neighbors)

            # Within support points
            average_dist = val[:, 0].sqrt()  # val.sqrt().mean(dim=1)
            ball = c*self.epsilon.squeeze()
            in_support = average_dist < ball

            # initiate features matrix
            features = torch.zeros(x.shape[0], self.modes).to(self.nodes.device)

            if in_support.sum() != 0:
                # Restrict domain
                val, idx_t = val[in_support], idx_t[in_support]

                # Extension Matrix
                val.div_(-4*self.epsilon.square()).exp_().div_(deg[idx_t]).div_(val.sum(dim=1).view(-1, 1))

                # Laplace/Fourier Features
                features[in_support] = val.unsqueeze(-1).mul(self.eigenvectors[idx_t]).sum(dim=1) * torch.pow(1 - self.epsilon.square()*self.eigenvalues, -1)

                # scale = average_dist[in_support].square().sub(ball.square()).pow(-1).mul(0.01).exp().div(ball.square().pow(-1).mul(-0.01).exp()).view(-1, 1)
                # features[in_support] = scale*weights*features[in_support] + (1-scale)*self.base_feature._featurize(x[in_support], normalize=normalize)
                features[in_support] = weights*features[in_support]

            # # Fourier Features
            # features[~in_support] = self.base_feature._featurize(x[~in_support], normalize=normalize)

            return features

    def scale_posterior(self, x, beta=1.0):
        # distance from manifold
        dist, _ = self.knn.search(x, 1)
        dist.sqrt_().squeeze_()

        # manifold support
        alpha = self.ball_scale*self.epsilon.squeeze()

        y = torch.zeros_like(dist)
        y[dist.abs() < alpha] = dist[dist.abs() < alpha].square().sub(alpha.square()).pow(-1).mul(beta).exp().div(alpha.square().pow(-1).mul(-beta).exp())
        return y


class LaplacianSymmetric(LinearOperator):
    def __init__(self, values, indices, epsilon, kernel):
        super(LaplacianSymmetric, self).__init__(values, indices, epsilon, kernel=kernel)

    def _matmul(self, x):
        return x.index_add(0, self._args[1][0, :], self._args[0].view(-1, 1) * x[self._args[1][1, :]], alpha=-1) \
                .index_add(0, self._args[1][1, :], self._args[0].view(-1, 1)*x[self._args[1][0, :]], alpha=-1).div(self._args[2].square())

    def _size(self):
        return torch.Size([self._kwargs['kernel'].nodes.shape[0], self._kwargs['kernel'].nodes.shape[0]])

    def _transpose_nonbatch(self):
        return self

    def diagonalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with gpytorch.settings.max_root_decomposition_size(3*self._kwargs['kernel'].modes):
            evals, evecs = super().diagonalization(method="lanczos")
            evals = evals[:self._kwargs['kernel'].modes]
            evecs = evecs.to_dense()[:, :self._kwargs['kernel'].modes]

        # from scipy.sparse import coo_matrix
        # from scipy.sparse.linalg import eigsh

        # # params
        # epsilon = self._kwargs['kernel'].epsilon[0]
        # num_points = self._kwargs['kernel'].nodes.shape[0]
        # device = self._kwargs['kernel'].nodes.device

        # # values & indices
        # opt = self._kwargs['kernel'].laplacian(operator='symmetric')
        # val = -opt._args[0]
        # idx = opt._args[1]

        # val = torch.cat((val.repeat(2).div(epsilon.square()), torch.ones(num_points).to(device).div(epsilon.square())), dim=0)
        # # val = torch.cat((val.repeat(2), torch.ones(num_points).to(device)), dim=0)
        # idx = torch.cat((idx, torch.stack((idx[1, :], idx[0, :]), dim=0), torch.arange(num_points).repeat(2, 1).to(device)), dim=1)
        # L = coo_matrix((val.detach().cpu().numpy(), (idx[0, :].cpu().numpy(), idx[1, :].cpu().numpy())), shape=(num_points, num_points))
        # T, V = eigsh(L, k=self._kwargs['kernel'].modes, which='SM')
        # evals = torch.from_numpy(T).float().to(device)
        # evecs = torch.from_numpy(V).float().to(device)

        return evals, evecs


class LaplacianRandomWalk(LinearOperator):
    def __init__(self, values, degree, indices, epsilon, kernel):
        super(LaplacianRandomWalk, self).__init__(values, degree, indices, epsilon, kernel=kernel)

    def _matmul(self, x):
        return x.index_add(0, self._args[2][0, :], self._args[0].view(-1, 1) * x[self._args[2][1, :]], alpha=-1).div(self._args[3].square())

    def _size(self):
        return torch.Size([self._kwargs['kernel'].nodes.shape[0], self._kwargs['kernel'].nodes.shape[0]])

    def _transpose_nonbatch(self):
        return self

    def diagonalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # evals, evecs = self._kwargs['kernel'].laplacian(operator='symmetric').diagonalization()
        # evecs = evecs.mul(self._args[1].sqrt().view(-1, 1))

        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import eigsh
        import scipy

        # params
        epsilon = self._kwargs['kernel'].epsilon[0]
        num_points = self._kwargs['kernel'].nodes.shape[0]
        device = self._kwargs['kernel'].nodes.device

        # values & indices
        opt = self._kwargs['kernel'].laplacian(operator='symmetric')
        val = -opt._args[0]
        idx = opt._args[1]

        val = torch.cat((val.repeat(2).div(epsilon.square()), torch.ones(num_points).to(device).div(epsilon.square())), dim=0)
        # val = torch.cat((val.repeat(2), torch.ones(num_points).to(device)), dim=0)
        idx = torch.cat((idx, torch.stack((idx[1, :], idx[0, :]), dim=0), torch.arange(num_points).repeat(2, 1).to(device)), dim=1)
        L = coo_matrix((val.detach().cpu().numpy(), (idx[0, :].cpu().numpy(), idx[1, :].cpu().numpy())), shape=(num_points, num_points))
        if self._kwargs['kernel'].modes == num_points:
            T, V = scipy.linalg.eigh(L.todense())
        else:
            T, V = eigsh(L, k=self._kwargs['kernel'].modes, which='SM')
        evals = torch.from_numpy(T).float().to(device)
        evecs = torch.from_numpy(V).float().to(device)
        evecs = evecs.mul(self._args[1].sqrt().view(-1, 1))

        return evals, evecs
