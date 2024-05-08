#!/usr/bin/env python
# encoding: utf-8

from abc import abstractmethod

import math
import torch
from torch import Tensor
import faiss
import faiss.contrib.torch_utils

import gpytorch
from gpytorch.priors import Prior
from gpytorch.constraints import Positive, Interval


from torch_geometric.utils import coalesce
from torch_geometric.nn import radius

from typing import Optional, Tuple, Union
from linear_operator import LinearOperator
from linear_operator.operators import LowRankRootLinearOperator, MatmulLinearOperator, RootLinearOperator

from manifold_gp.priors.inverse_gamma_prior import InverseGammaPrior

from ..utils import NearestNeighbors, bump_function
from ..operators import GraphLaplacianOperator


class RiemannKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self,
                 x: torch.Tensor,
                 nearest_neighbors: Optional[int] = 10,
                 laplacian_normalization: Optional[str] = "symmetric",
                 num_modes: Optional[int] = 100,
                 bump_scale: Optional[float] = 1.0,
                 bump_decay: Optional[float] = 0.01,
                 graphbandwidth_prior: Optional[Prior] = None,
                 graphbandwidth_constraint: Optional[Interval] = None,
                 **kwargs):
        super(RiemannKernel, self).__init__(**kwargs)

        self.knn = NearestNeighbors(x, nlist=1)
        self.nearest_neighbors = nearest_neighbors
        self.edge_index, self.edge_value = self.knn.graph(self.nearest_neighbors, nprobe=1)
        self.laplacian_normalization = laplacian_normalization
        self.num_modes = num_modes
        self.bump_scale = bump_scale
        self.bump_decay = bump_decay

        if graphbandwidth_constraint is None:
            graphbandwidth_constraint = Positive()

        self.register_parameter(
            name='raw_graphbandwidth',
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )

        if graphbandwidth_prior is not None:
            if not isinstance(graphbandwidth_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(graphbandwidth_prior).__name__)
            self.register_prior(
                "graphbandwidth_prior", graphbandwidth_prior, self._graphbandwidth_param, self._graphbandwidth_closure
            )

        self.register_constraint("raw_graphbandwidth", graphbandwidth_constraint)

    def _graphbandwidth_param(self, m) -> Tensor:
        # Used by the graphbandwidth_prior
        return m.graphbandwidth

    def _graphbandwidth_closure(self, m, v: Tensor) -> Tensor:
        # Used by the graphbandwidth_prior
        return m._set_graphbandwidth(v)

    def _set_graphbandwidth(self, value: Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_graphbandwidth)

        self.initialize(raw_graphbandwidth=self.raw_graphbandwidth_constraint.inverse_transform(value))

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **kwargs) -> Tensor:
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)
        z1 = self.features(x1)
        if not x1_eq_x2:
            z2 = self.features(x2)
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

    @abstractmethod
    def spectral_density(self):
        raise NotImplementedError()

    @property
    def graphbandwidth(self) -> Tensor:
        return self.raw_graphbandwidth_constraint.transform(self.raw_graphbandwidth)

    @graphbandwidth.setter
    def graphbandwidth(self, value: Tensor):
        self._set_graphbandwidth(value)

    def laplacian(self):
        return GraphLaplacianOperator(self.edge_value, self.edge_index, self.knn.x.shape[0], self.graphbandwidth, self.laplacian_normalization)

    def eval(self):
        self.laplacian_operator = self.laplacian()
        with torch.no_grad():
            self.eigval, self.eigvec = self.laplacian_operator.diagonalization(num_modes=self.num_modes)

        return super().eval()

    def features(self, x: Tensor) -> Tensor:
        if torch.equal(x, self.knn.x):
            spectral_density = self.spectral_density()
            spectral_density /= spectral_density.sum()
            return (spectral_density * self.eigvec.shape[0]).sqrt() * self.eigvec
        else:
            edge_value, edge_index = self.knn.search(x, self.nearest_neighbors)
            x_within_support = edge_value[:, 0].sqrt() < self.bump_scale*self.graphbandwidth.squeeze()
            features = torch.zeros(x.shape[0], self.num_modes, device=x.device)

            if x_within_support.sum() != 0:
                spectral_density = self.spectral_density().div((1 - self.graphbandwidth.square() * self.eigval).square())
                spectral_density /= spectral_density.sum()
                spectral_density *= self.knn.x.shape[0]
                features[x_within_support] = spectral_density.sqrt() * self.laplacian_operator.out_of_sample(self.eigvec, edge_value[x_within_support], edge_index[x_within_support]) * \
                    bump_function(edge_value[x_within_support, 0].sqrt(), self.bump_scale*self.graphbandwidth.squeeze(), self.bump_decay).unsqueeze(-1)

            return features
