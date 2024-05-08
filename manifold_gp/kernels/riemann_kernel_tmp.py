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


class RiemannKernelTmp(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self,
                 nodes: torch.Tensor,
                 neighbors: Optional[int] = 2,
                 operator: Optional[str] = "randomwalk",
                 method: Optional[str] = "lanczos",
                 modes: Optional[int] = 10,
                 bump_scale: Optional[float] = 1.0,
                 bump_decay: Optional[float] = 0.01,
                 prior_bandwidth: Optional[bool] = False,
                 **kwargs):
        super(RiemannKernelTmp, self).__init__(**kwargs)

        self.nodes = nodes  # remove this

        # adjust data dimension (we removed it for the moment because minimum embedding dimension is 2)
        # nodes = nodes.unsqueeze(-1) if nodes.ndimension() == 1 else nodes

        # data info
        self.memory_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.memory_device = torch.device("cpu")
        self.num_samples, self.num_dims = nodes.shape

        # Number of neighborhoods for KKN
        self.neighbors = neighbors

        # Type of Laplacian normalization (randomwalk, symmetric, unnormalized)
        self.operator = operator

        # Method used for eigenvalue decomposition (lanczos, arpack, exact)
        self.method = method

        # Number of modes to extract
        self.modes = modes

        # Support and decay of the bump function
        self.bump_scale = bump_scale
        self.bump_decay = bump_decay

        # Enable prior on the graph bandwidth
        self.prior_bandwidth = prior_bandwidth

        # KNN Faiss
        if self.memory_device.type == 'cuda':
            res = faiss.StandardGpuResources()
            self.knn = faiss.GpuIndexIVFFlat(res, self.num_dims, 1, faiss.METRIC_L2)
        else:
            self.knn = faiss.IndexFlatL2(self.num_dims)

        # Build graph (in addition calculates the necessary parameters to initialize the graph bandwidth prior)
        self.generate_graph(nodes)

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
        z1 = self.features(x1, c=self.bump_scale)
        if not x1_eq_x2:
            z2 = self.features(x2, c=self.bump_scale)
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

    def generate_graph(self, nodes):
        # KNN Search
        self.knn.reset()
        self.knn.train(nodes)
        self.knn.add(nodes)
        val, idx = self.knn.search(nodes, self.neighbors)

        # Calculate epsilon lower bound (99% kernel decay)
        min_keval = torch.tensor(1e-4).to(self.memory_device)
        self.eps_gte = val[:, 1].max().div(-4*min_keval.log()).sqrt()

        # Gamma distribution hyperparameters
        if self.prior_bandwidth:
            p_50 = val[:, 1:].sqrt().mean(dim=1).sort()[0][int(round(val.shape[0]*0.50))]
            k_std = 4*p_50/(p_50-self.eps_gte)**2
            self.eps_rate = k_std
            self.eps_concentration = k_std * p_50 + 1

        # Make KNN Symmetric
        rows = torch.arange(idx.shape[0]).repeat_interleave(idx.shape[1]-1).to(self.memory_device)
        cols = idx[:, 1:].reshape(1, -1).squeeze()
        val = val[:, 1:].reshape(1, -1).squeeze()
        # rows = torch.arange(idx.shape[0]).repeat_interleave(idx.shape[1]).to(self.nodes.device)
        # cols = idx.reshape(1, -1).squeeze()
        # val = val.reshape(1, -1).squeeze()
        split = cols > rows
        rows, cols = torch.cat([rows[split], cols[~split]], dim=0), torch.cat([cols[split], rows[~split]], dim=0)
        idx = torch.stack([rows, cols], dim=0)
        val = torch.cat([val[split], val[~split]])
        self.indices, self.values = coalesce(idx, val, reduce='mean')  # only function that requires torch_geometric (replace it with pytorch)

    # def laplacian(self, operator):
    #     if operator == 'symmetric':
    #         # Adjacency Matrix
    #         idx = self.indices
    #         val = self.values.div(-4*self.epsilon.square()).exp().squeeze()

    #         # Diffusion Maps Normalization
    #         deg = torch.zeros(self.num_samples).to(self.memory_device).scatter_add_(0, idx[0, :], val).scatter_add_(0, idx[1, :], val)
    #         val = val.div(deg[idx[0, :]]*deg[idx[1, :]])

    #         # Symmetric Laplacian Normalization
    #         deg = torch.zeros(self.num_samples).to(self.memory_device).scatter_add_(0, idx[0, :], val).scatter_add(0, idx[1, :], val).sqrt()
    #         val.div_(deg[idx[0, :]]*deg[idx[1, :]])

    #         return LaplacianSymmetric(val, idx, self.epsilon, self)
    #     elif operator == 'randomwalk':
    #         # Adjacency Matrix
    #         val = self.values.div(-4*self.epsilon.square()).exp().squeeze().repeat(2)

    #         # Add diagonal and build indices
    #         val = torch.cat([torch.ones(self.num_samples).to(self.memory_device), val])
    #         idx = torch.cat((torch.arange(self.num_samples).repeat(2, 1).to(self.memory_device), self.indices, torch.stack((self.indices[1, :], self.indices[0, :]), dim=0)), dim=1)
    #         # idx = torch.cat((self.indices, torch.stack((self.indices[1, :], self.indices[0, :]), dim=0)), dim=1)

    #         # Diffusion Maps Normalization
    #         deg = torch.zeros(self.num_samples).to(self.memory_device).scatter_add_(0, idx[0, :], val)
    #         val = val.div(deg[idx[0, :]]*deg[idx[1, :]])

    #         # Random Walk Normalization
    #         deg = torch.zeros(self.num_samples).to(self.memory_device).scatter_add_(0, idx[0, :], val)
    #         val.div_(deg[idx[0, :]])

    #         return LaplacianRandomWalk(val, deg.pow(-1), idx, self.epsilon, self)

    def laplacian(self, operator):
        # build indices
        idx = torch.cat((torch.arange(self.num_samples).repeat(2, 1).to(self.memory_device),
                         self.indices,
                         torch.stack((self.indices[1, :], self.indices[0, :]), dim=0)), dim=1)

        # build values
        val = torch.cat((torch.zeros(self.num_samples).to(self.memory_device),
                         self.values.repeat(2)))

        # adjacency matrix
        val = val.div(-4*self.epsilon.square()).exp().squeeze()

        # diffusion maps normalization
        deg = torch.zeros(self.num_samples).to(self.memory_device).scatter_add_(0, idx[0, :], val)
        val = val.div(deg[idx[0, :]]*deg[idx[1, :]])

        # symmetric laplacian normalization
        if operator == 'symmetric':
            deg = torch.zeros(self.num_samples).to(self.memory_device).scatter_add_(0, idx[0, :], val).sqrt()
            val.div_(deg[idx[0, :]]*deg[idx[1, :]])
            return LaplacianSymmetric(val, idx, self.epsilon, self)
        # random walk laplacian normalization
        elif operator == 'randomwalk':
            deg = torch.zeros(self.num_samples).to(self.memory_device).scatter_add_(0, idx[0, :], val)
            val.div_(deg[idx[0, :]])
            return LaplacianRandomWalk(val, deg.pow(-1), idx, self.epsilon, self)

    def generate_eigenpairs(self):
        # evals, evecs = self.laplacian(operator=self.operator).diagonalization()
        # self.eigenvalues = evals.detach()
        # self.eigenvectors = evecs.detach()
        with torch.no_grad(), gpytorch.settings.cg_tolerance(10000):
            self.eigenvalues, self.eigenvectors = self.laplacian(operator=self.operator).diagonalization()

    def features(self, x: Tensor, c: float = 1.0) -> Tensor:
        if torch.equal(x, self.nodes):  # check how to asses if I am evaluating on the training data (self.nodes not available anymore)
            # Spectral density
            s = self.spectral_density()

            # Normalization
            if self.operator == "randomwalk":
                s_norm = (s*self.eigenvectors.square()).sum()
                s = s/s_norm
                # s /= (s*self.eigenvectors.square()).sum()
            elif self.operator == "symmetric":
                s_norm = s.sum()
                s = s/s_norm
                # s /= s.sum()

            return (s * self.num_samples).sqrt() * self.eigenvectors
        else:
            # KNN Search
            val_test, idx_test = self.knn.search(x, self.neighbors)

            # Within support points
            average_dist = val_test[:, 0].sqrt()  # val.sqrt().mean(dim=1)
            ball = c*self.epsilon.squeeze()
            in_support = average_dist < ball

            # initiate features matrix
            features = torch.zeros(x.shape[0], self.modes).to(self.memory_device)

            if in_support.sum() != 0:
                scale_epsilon = 1.0

                # Degree Matrix Train Set (this part does not have to be done online)
                val_train = torch.cat([torch.ones(self.num_samples).to(self.memory_device), self.values.div(-4*(self.epsilon*1.0).square()).exp().squeeze().repeat(2)])
                idx_train = torch.cat((torch.arange(self.num_samples).repeat(2, 1).to(self.memory_device), self.indices, torch.stack((self.indices[1, :], self.indices[0, :]), dim=0)), dim=1)
                deg_train = torch.zeros(self.num_samples).to(self.memory_device).scatter_add_(0, idx_train[0, :], val_train)

                # Restrict domain and calculate extension matrix up to first normalization
                val_test, idx_test = val_test[in_support], idx_test[in_support]  # eps = (self.epsilon/3).square()
                deg_test = val_test.sum(dim=1)
                val_test.div_(-4*(self.epsilon*scale_epsilon).square()).exp_().div_(deg_train[idx_test]).div_(deg_test.view(-1, 1))

                # Spectral density
                s = self.spectral_density().div((1 - (self.epsilon*scale_epsilon).square() * self.eigenvalues).square())

                # Extension Matrix
                if self.operator == "randomwalk":
                    s = s/(s*self.eigenvectors.square()).sum()*self.num_samples
                    val_test.div_(val_test.sum(dim=1).view(-1, 1))
                elif self.operator == "symmetric":
                    s = s/s.sum()*self.num_samples
                    deg_test = val_test.sum(dim=1).sqrt()
                    val_test.div_(deg_test[idx_test]).div_(deg_test.view(-1, 1))

                # Laplace/Fourier Features
                # features[in_support] = val_test.unsqueeze(-1).mul(self.eigenvectors[idx_test]).sum(dim=1)
                # features[in_support] = features[in_support].div(features[in_support].norm(dim=0)).mul(self.eigenvectors.norm(dim=0))
                # features[in_support] = s.sqrt() * features[in_support]
                features[in_support] = s.sqrt() * val_test.unsqueeze(-1).mul(self.eigenvectors[idx_test]).sum(dim=1)
                # features[in_support] = val_test.unsqueeze(-1).mul(self.eigenvectors[idx_test]).sum(dim=1).div(1 - (self.epsilon*scale_epsilon).square() * self.eigenvalues)

            return features  # self.bump_function(x).unsqueeze(-1)*

    def bump_function(self, x):
        # distance from manifold
        dist, _ = self.knn.search(x, 1)
        dist.sqrt_().squeeze_()

        # manifold support
        alpha = self.bump_scale*self.epsilon.squeeze()
        beta = self.bump_decay

        y = torch.zeros_like(dist)
        y[dist.abs() < alpha] = dist[dist.abs() < alpha].square().sub(alpha.square()).pow(-1).mul(beta).exp().div(alpha.square().pow(-1).mul(-beta).exp())
        return y


class LaplacianSymmetric(LinearOperator):
    def __init__(self, values, indices, epsilon, kernel):
        super(LaplacianSymmetric, self).__init__(values, indices, epsilon, kernel=kernel)

    def _matmul(self, x):
        return x.index_add(0, self._args[1][0, :], self._args[0].view(-1, 1) * x[self._args[1][1, :]], alpha=-1).div(self._args[2].square())
        # return x.index_add(0, self._args[1][0, :], self._args[0].view(-1, 1) * x[self._args[1][1, :]], alpha=-1) \
        #         .index_add(0, self._args[1][1, :], self._args[0].view(-1, 1)*x[self._args[1][0, :]], alpha=-1).div(self._args[2].square())

    def _size(self):
        return torch.Size([self._kwargs['kernel'].num_samples, self._kwargs['kernel'].num_samples])

    def _transpose_nonbatch(self):
        return self

    def diagonalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._kwargs['kernel'].method == 'lanczos':
            print('lanczos')
            num_points = self._kwargs['kernel'].num_samples
            num_eigs = 5*self._kwargs['kernel'].modes

            with gpytorch.settings.max_root_decomposition_size(num_eigs if num_eigs <= num_points else num_points):
                evals, evecs = super().diagonalization(method="lanczos")
                evals.data[0] = 0.0000e+00  # sometimes the first eigenvalues is 1.0 for some reason
                evals = evals[:self._kwargs['kernel'].modes]
                evecs = evecs.to_dense()[:, :self._kwargs['kernel'].modes]
        elif self._kwargs['kernel'].method == 'arpack':
            print('arpack')
            from scipy.sparse import coo_matrix
            from scipy.sparse.linalg import eigsh

            # params
            epsilon = self._kwargs['kernel'].epsilon[0]
            num_points = self._kwargs['kernel'].num_samples
            device = self._kwargs['kernel'].memory_device

            # values & indices
            opt = self._kwargs['kernel'].laplacian(operator='symmetric')
            val = -opt._args[0]
            val[:num_points] += 1.0
            val.div_(epsilon.square())
            idx = opt._args[1]

            # val = torch.cat((val.repeat(2).div(epsilon.square()), torch.ones(num_points).to(device).div(epsilon.square())), dim=0)
            # val = torch.cat((val.repeat(2), torch.ones(num_points).to(device)), dim=0)
            # idx = torch.cat((idx, torch.stack((idx[1, :], idx[0, :]), dim=0), torch.arange(num_points).repeat(2, 1).to(device)), dim=1)
            laplacian = coo_matrix((val.detach().cpu().numpy(), (idx[0, :].cpu().numpy(), idx[1, :].cpu().numpy())), shape=(num_points, num_points))
            evals, evecs = eigsh(laplacian, k=self._kwargs['kernel'].modes, which='SM')
            evals = torch.from_numpy(evals).float().to(device)
            evecs = torch.from_numpy(evecs).float().to(device)
        elif self._kwargs['kernel'].method == 'exact':
            print('exact')
            # params
            epsilon = self._kwargs['kernel'].epsilon[0]
            num_points = self._kwargs['kernel'].num_samples
            device = self._kwargs['kernel'].memory_device

            # values & indices
            opt = self._kwargs['kernel'].laplacian(operator='symmetric')
            val = -opt._args[0]
            val[:num_points] += 1.0
            val.div_(epsilon.square())
            idx = opt._args[1]

            # val = torch.cat((val.repeat(2).div(epsilon.square()), torch.ones(num_points).to(device).div(epsilon.square())), dim=0)
            # idx = torch.cat((idx, torch.stack((idx[1, :], idx[0, :]), dim=0), torch.arange(num_points).repeat(2, 1).to(device)), dim=1)
            laplacian = torch.sparse_coo_tensor(idx, val, [num_points, num_points]).to_dense()  # dtype=torch.float64
            evals, evecs = torch.linalg.eigh(laplacian)

        return evals[:self._kwargs['kernel'].modes], evecs[:, :self._kwargs['kernel'].modes]


class LaplacianRandomWalk(LinearOperator):
    def __init__(self, values, degree, indices, epsilon, kernel):
        super(LaplacianRandomWalk, self).__init__(values, degree, indices, epsilon, kernel=kernel)

    def _matmul(self, x):
        return x.index_add(0, self._args[2][0, :], self._args[0].view(-1, 1) * x[self._args[2][1, :]], alpha=-1).div(self._args[3].square())

    def _size(self):
        return torch.Size([self._kwargs['kernel'].num_samples, self._kwargs['kernel'].num_samples])

    def _transpose_nonbatch(self):
        return self

    def diagonalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        evals, evecs = self._kwargs['kernel'].laplacian(operator='symmetric').diagonalization()
        evecs = evecs.mul(self._args[1].sqrt().view(-1, 1))

        return evals, evecs
