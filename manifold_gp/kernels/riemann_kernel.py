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
                 support_kernel: Optional[gpytorch.kernels.Kernel] = gpytorch.kernels.RBFKernel(),
                 epsilon_prior: Optional[gpytorch.priors.Prior] = None,
                 **kwargs):
        super(RiemannKernel, self).__init__(**kwargs)

        # Hyperparameter
        self.nodes = nodes.unsqueeze(-1) if nodes.ndimension() == 1 else nodes
        self.neighbors = neighbors
        self.operator = operator
        self.modes = 2*modes
        self.ball_scale = ball_scale
        self.support_kernel = support_kernel

        self.base_feature = gpytorch.kernels.RFFKernel(modes)
        self.base_feature._init_weights(self.nodes.size(-1), self.base_feature.num_samples)

        # KNN Faiss
        if self.nodes.is_cuda:
            res = faiss.StandardGpuResources()
            self.knn = faiss.GpuIndexIVFFlat(res, self.nodes.shape[1], 1, faiss.METRIC_L2)
            # res = faiss.StandardGpuResources()
            # self.knn = faiss.GpuIndexFlatL2(res, self.nodes.shape[1])
        else:
            self.knn = faiss.IndexFlatL2(self.nodes.shape[1])

        # Build graph
        self.generate_graph()

        # Heat kernel length parameter for Laplacian approximation
        self.register_parameter(
            name='raw_epsilon', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1).to(nodes.device), requires_grad=True)
        )

        if epsilon_prior is not None:
            if not isinstance(epsilon_prior, gpytorch.priors.Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(epsilon_prior).__name__)
            self.register_prior(
                "epsilon_prior", epsilon_prior, self._epsilon_param, self._epsilon_closure
            )

        # self.register_prior(
        #     "epsilon_prior", InverseGammaPrior(self.eps_concentration, self.eps_rate), self._epsilon_param, self._epsilon_closure
        # )

        # self.register_prior(
        #     "epsilon_prior", gpytorch.priors.GammaPrior(self.eps_concentration, self.eps_rate), self._epsilon_param, self._epsilon_closure
        # )

        # self.register_constraint("raw_epsilon", GreaterThan(self.eps_gte))
        self.register_constraint("raw_epsilon", Positive())

    @abstractmethod
    def spectral_density(self):
        raise NotImplementedError()

    # def base(self, x1, x2, diag=False):
    #     if diag:
    #         if torch.equal(x1, x2):
    #             return torch.sum(self.spectral_density().T*self.eigenfunctions(x1).pow(2), dim=0)
    #         else:
    #             return torch.sum(self.spectral_density().T*self.eigenfunctions(x1)*self.eigenfunctions(x2), dim=0)
    #     else:
    #         return torch.mm(self.eigenfunctions(x1).T, self.spectral_density().T*self.eigenfunctions(x2))

    # def forward(self, x1, x2, diag=False, **params):
    #     if diag:
    #         if torch.equal(x1, x2):
    #             return self.variance(x1)
    #         else:
    #             return self.base(x1, x2, diag=True) + self.support_kernel.forward(x1, x2, diag=True)*torch.sqrt(self.variance(x1) - self.base(x1, x1, diag=True))*torch.sqrt(self.variance(x2) - self.base(x2, x2, diag=True))
    #     else:
    #         return self.base(x1, x2) + self.support_kernel.forward(x1, x2)*torch.outer(torch.sqrt(self.variance(x1) - self.base(x1, x1, diag=True)), torch.sqrt(self.variance(x2) - self.base(x2, x2, diag=True)))

    # def temp_forward(self, x1, x2, diag=False, **params):
    #     if diag:
    #         if torch.equal(x1, x2):
    #             return torch.sum(self.features(x1).pow(2), dim=0)
    #         else:
    #             return torch.sum(self.features(x1)*self.features(x2), dim=0)
    #     else:
    #         return torch.mm(self.features(x1), self.features(x2).T)

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

        D = 1  # float(self.modes)
        if diag:
            return (z1 * z2).sum(-1) / D
        if x1_eq_x2:
            # Exploit low rank structure, if there are fewer features than data points
            if z1.size(-1) < z2.size(-2):
                return LowRankRootLinearOperator(z1 / math.sqrt(D))
            else:
                return RootLinearOperator(z1 / math.sqrt(D))
        else:
            return MatmulLinearOperator(z1 / D, z2.transpose(-1, -2))

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

        # Calculate epsilon lower constraint
        min_weight = torch.tensor(1e-4).to(self.nodes.device)
        self.eps_gte = val[:, 1].max().div(-2*min_weight.log()).sqrt()

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

        # # Calculate epsilon prior parameter (Inverse Gamma)
        # self.eps_concentration = self.values.sqrt().std().pow(-1)*100.0
        # self.eps_rate = (self.eps_concentration+1)*self.values.sort()[0][int(self.values.shape[0]*0.99)].sqrt()

        # # Gamma distribution
        # self.eps_concentration = 100.0*self.values.sqrt().std().pow(-1) * self.values.sort()[0][int(self.values.shape[0]*0.99)].sqrt() + 1
        # self.eps_rate = 100.0*self.values.sqrt().std().pow(-1)

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

    # def eigenfunctions(self, x):
    #     distances, indices = self.knn.search(x, self.neighbors+1)  # self.neighbors or fix it
    #     # distances[distances > self.epsilon.square()] = 0
    #     # return torch.sum(self.eigenvectors[indices].permute(2, 0, 1) * distances.div(-2*self.epsilon.square()).exp(), dim=2)
    #     return self.inverse_distance_weighting(distances, self.eigenvectors[indices].permute(2, 0, 1))

    # def features(self, x: Tensor, normalize: bool = False, c: float = 1.0) -> Tensor:
    #     # features weights
    #     weights = (self.spectral_density() / self.normalization).sqrt()

    #     if torch.equal(x, self.nodes):
    #         return weights * self.eigenvectors
    #     else:
    #         z = weights * self.laplace_features(x)  # * math.sqrt(self.nodes.shape[0])
    #         # test_idx, train_idx = radius(self.nodes, x, c*self.epsilon, max_num_neighbors=self.nodes.shape[0])

    #         # # if not train point inside r-ball of query points return rff
    #         # if test_idx.numel() == 0:
    #         #     # return self.base_feature._featurize(x, normalize)
    #         #     return weights * self.eigenvectors.mean(dim=0).repeat(x.shape[0], 1)

    #         # # calculate laplace features
    #         # else:
    #         #     z = weights * self.inverse_distance_weighting(x, self.eigenvectors, train_idx, test_idx)  # * math.sqrt(self.nodes.shape[0])

    #         #     laplace_idx = (torch.arange(x.size(0)).to(self.nodes.device).unsqueeze(-1) == torch.unique(test_idx)).any(1)
    #         #     # base_idx = torch.arange(x.size(0)).to(self.nodes.device) != torch.unique(test_idx)

    #         #     # z[~laplace_idx] = self.base_feature._featurize(x[~laplace_idx], normalize)
    #         #     z[~laplace_idx] = weights * self.eigenvectors.mean(dim=0).unsqueeze(0)

    #         return z

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

    # def eigenfunctions(self, x, c=1.0):
    #     # test_idx, train_idx = radius(self.nodes, x, c*self.epsilon)
    #     # if test_idx.numel() == 0:
    #     #     return torch.zeros(self.modes, x.shape[0])

    #     # val = (x[test_idx, :] - self.nodes[train_idx, :]).square().sum(dim=1).pow(-1)
    #     # norm = torch.zeros(x.shape[0]).to(self.nodes.device).scatter_add_(0, test_idx, val)
    #     # weigths = torch.zeros(x.shape[0], self.modes).to(self.nodes.device).index_add_(0, test_idx, self.eigenvectors[train_idx, :]*val.view(-1, 1))
    #     # nz_idx = norm >= 1e-6
    #     # weigths[nz_idx] /= norm[nz_idx]

    #     # return weigths.T
    #     return self.locally_weighted_regression(x, self.eigenvectors, c*self.epsilon)

    def inverse_distance_weighting(self, x, weights, train_idx, test_idx):
        # distances
        values = (x[test_idx, :] - self.nodes[train_idx, :]).square().sum(dim=1)

        # calculate non-zero values (zero values correspond to points that coincide with points from the training set)
        nz_values = values >= 1e-6
        values[nz_values] = values[nz_values].pow(-1)

        # calculate inverse (squared) distance weighting (this can be become an external function) and the normalization terms
        norms = torch.zeros(x.shape[0]).to(self.nodes.device).scatter_add_(0, test_idx, values)
        values = torch.zeros(x.shape[0], weights.shape[1]).to(self.nodes.device).index_add_(0, test_idx, weights[train_idx, :]*values.view(-1, 1))

        # calculate non-zero norms (this comes from isolated points, e.g. no neighborhood within the query ball)
        nz_norms = norms >= 1e-6
        values[nz_norms] /= norms[nz_norms].view(-1, 1)

        # assign to the zeros values the weight at a specific point from training set
        values[test_idx[~nz_values]] = weights[train_idx[~nz_values]]

        return values

    def locally_weighted_regression(self, x, weights, train_idx, test_idx, s=1.0):
        # distances
        values = (x[test_idx, :] - self.nodes[train_idx, :]).square().sum(dim=1)

        # exp
        values = values.div(-4*s**2).exp()

        # calculate inverse (squared) distance weighting (this can be become an external function) and the normalization terms
        norms = torch.zeros(x.shape[0]).to(self.nodes.device).scatter_add_(0, test_idx, values)
        values = torch.zeros(x.shape[0], weights.shape[1]).to(self.nodes.device).index_add_(0, test_idx, weights[train_idx, :]*values.view(-1, 1))

        # # calculate non-zero norms (this comes from isolated points, e.g. no neighborhood within the query ball)
        # nz_norms = norms >= 1e-6
        # values[nz_norms] /= norms[nz_norms].view(-1, 1)
        values /= norms.view(-1, 1)

        return values

    def laplace_features(self, x):
        # Degree Matrix Train Set (this part does not have to be done online)
        idx = torch.cat((self.indices, torch.stack((self.indices[1, :], self.indices[0, :]), dim=0)), dim=1)
        deg = torch.zeros(self.nodes.shape[0]).to(self.nodes.device).scatter_add_(0, idx[0, :], self.values.div(-4*self.epsilon.square()).exp().squeeze().repeat(2))

        # Extension Matrix
        val, idx_t = self.knn.search(x, self.neighbors)
        val.div_(-4*self.epsilon.square()).exp_().div_(deg[idx_t]).div_(val.sum(dim=1).view(-1, 1))

        # Feature
        return val.unsqueeze(-1).mul(self.eigenvectors[idx_t]).sum(dim=1) * torch.pow(1 - self.epsilon.square()*self.eigenvalues, -1)

    def nystrom_formula(self, test_x, train_x, weights, s=1.0):
        l = -.25 / s**2
        xx = torch.einsum('ij,ij->i', test_x, test_x).unsqueeze(1)
        yy = torch.einsum('ij,ij->i', train_x, train_x).unsqueeze(0)
        k = -2 * torch.mm(test_x, train_x.T) + xx + yy
        k *= l
        k = torch.exp(k)
        return torch.mm(k, weights)  # .div(k.sum(dim=1).unsqueeze(-1))

    def _sigmoid_function(self, x, alpha, beta):
        return x.sub(beta).mul(-alpha).exp().add(1.0).pow(-1)

    def _bump_function(self, x, alpha, beta=1.0):
        # y = torch.zeros_like(x)
        # y[x.abs() < alpha] = x[x.abs() < alpha].square().sub(alpha.square()).pow(-1).mul(beta).exp().div(alpha.square().pow(-1).mul(-beta).exp())
        # return y
        return x.square().sub(alpha.square()).pow(-1).mul(beta).exp().div(alpha.square().pow(-1).mul(-beta).exp())

    def scale_posterior(self, x, beta=1.0):
        # distance from manifold
        dist, _ = self.knn.search(x, 1)
        dist.sqrt_().squeeze_()

        # manifold support
        alpha = self.ball_scale*self.epsilon.squeeze()

        y = torch.zeros_like(dist)
        y[dist.abs() < alpha] = dist[dist.abs() < alpha].square().sub(alpha.square()).pow(-1).mul(beta).exp().div(alpha.square().pow(-1).mul(-beta).exp())
        return y

    # def variance(self, x):
    #     distances, indices = self.knn.search(x, self.neighbors)

    #     return torch.sum(self.spectral_density().T*self.inverse_distance_weighting(distances, self.eigenvectors[indices].pow(2).permute(2, 0, 1)), dim=0)

    # def variance(self, x, c=1.0):
    #     weights = torch.sum(self.spectral_density()*self.eigenvectors.pow(2), dim=1)
    #     # return self.locally_weighted_regression(x, weights.view(-1, 1), c*self.epsilon)
    #     return weights

    # def inverse_distance_weighting(self, d, y):
    #     u = torch.zeros(y.shape[0], y.shape[1]).to(d.device)
    #     idx_zero = torch.any(d <= 1e-8, 1)
    #     u[:, idx_zero] = y[:, idx_zero, 0]
    #     u[:, ~idx_zero] = torch.sum(
    #         y[:, ~idx_zero, :] / d[~idx_zero, :], dim=2).div(torch.sum(d[~idx_zero, :].pow(-1), dim=1))
    #     return u


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
        # with gpytorch.settings.max_root_decomposition_size(3*self._kwargs['kernel'].modes):
        #     evals, evecs = super().diagonalization(method="lanczos")
        #     evals = evals[:self._kwargs['kernel'].modes]
        #     evecs = evecs.to_dense()[:, :self._kwargs['kernel'].modes]

        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import eigsh

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
        T, V = eigsh(L, k=self._kwargs['kernel'].modes, which='SM')
        evals = torch.from_numpy(T).float().to(device)
        evecs = torch.from_numpy(V).float().to(device)

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
        T, V = eigsh(L, k=self._kwargs['kernel'].modes, which='SM')
        evals = torch.from_numpy(T).float().to(device)
        evecs = torch.from_numpy(V).float().to(device)
        evecs = evecs.mul(self._args[1].sqrt().view(-1, 1))

        return evals, evecs
