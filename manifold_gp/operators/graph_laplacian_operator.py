#!/usr/bin/env python
# encoding: utf-8

from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import gpytorch

from jaxtyping import Float, Int32
from torch import Tensor

from linear_operator.operators._linear_operator import LinearOperator, to_dense

from linear_operator.utils.broadcasting import _pad_with_singletons
from linear_operator.utils.getitem import _noop_index, IndexType
from linear_operator.utils.memoize import cached

from torch_sparse import transpose, spmm

from torch.nn.functional import normalize


class GraphLaplacianOperator(LinearOperator):
    def __init__(
            self,
            x: Float[Tensor, "N"],
            idx: Int32[Tensor, "N"],
            operator_dimension: int,
            graphbandwidth: Tensor,
            normalization: Optional[str] = "randomwalk",  # "symmetric"
            self_loops: Optional[bool] = True,
            transposed: Optional[bool] = False
    ):
        super().__init__(
            x,
            idx=idx,
            operator_dimension=operator_dimension,
            graphbandwidth=graphbandwidth,
            normalization=normalization,
            self_loops=self_loops,
            transposed=transposed
        )
        self.x = x
        self.idx = idx
        self.operator_dimension = operator_dimension
        self.graphbandwidth = graphbandwidth
        self.normalization = normalization
        self.self_loops = self_loops
        self.transposed = transposed

    @property
    @cached(name="adjacency_unnorm_mat")
    def adjacency_unnorm_mat(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "M"]:
        # print("adjacency_unnorm_mat")
        return self.x.div(-4*self.graphbandwidth.square()).exp().squeeze()

    @property
    @cached(name="degree_unnorm_mat")
    def degree_unnorm_mat(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "N"]:
        # print("degree_unnorm_mat")
        if self.self_loops:
            return torch.ones(self.operator_dimension, device=self.x.device) \
                .scatter_add_(0, self.idx[0, :], self.adjacency_unnorm_mat) \
                .scatter_add_(0, self.idx[1, :], self.adjacency_unnorm_mat)
        else:
            return torch.zeros(self.operator_dimension, device=self.x.device) \
                .scatter_add_(0, self.idx[0, :], self.adjacency_unnorm_mat) \
                .scatter_add_(0, self.idx[1, :], self.adjacency_unnorm_mat)

    @property
    @cached(name="adjacency_mat")
    def adjacency_mat(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "M"]:
        # print("adjacency_mat")
        return self.adjacency_unnorm_mat.div(self.degree_unnorm_mat[self.idx[0, :]]*self.degree_unnorm_mat[self.idx[1, :]])

    @property
    @cached(name="degree_mat")
    def degree_mat(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "N"]:
        # print("degree_mat")
        if self.self_loops:
            return self.degree_unnorm_mat.pow(-2) \
                .scatter_add_(0, self.idx[0, :], self.adjacency_mat) \
                .scatter_add(0, self.idx[1, :], self.adjacency_mat)
        else:
            return torch.zeros(self.operator_dimension, device=self.x.device) \
                .scatter_add_(0, self.idx[0, :], self.adjacency_mat) \
                .scatter_add(0, self.idx[1, :], self.adjacency_mat)

    @property
    @cached(name="laplacian_diag")
    def laplacian_diag(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "N"]:
        # print("laplacian_diag")
        if self.self_loops:
            return (1 - self.degree_unnorm_mat.pow(-2)*self.degree_mat.pow(-1)).div(self.graphbandwidth.square().squeeze())
        else:
            return torch.ones(self.operator_dimension, device=self.x.device).div(self.graphbandwidth.square().squeeze())

    def _diagonal(self: LinearOperator) -> Tensor:
        return self.laplacian_diag

    @property
    @cached(name="laplacian_triu")
    def laplacian_triu(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "M"]:
        degree_sqrt = self.degree_mat.sqrt()
        return self.adjacency_mat.div(degree_sqrt[self.idx[0, :]]*degree_sqrt[self.idx[1, :]]).div(self.graphbandwidth.square().squeeze())

    def _matmul(
        self: Float[LinearOperator, "N N"],
        rhs: Union[Float[torch.Tensor, "N C"], Float[torch.Tensor, "N"]],
    ):
        if self.normalization == 'randomwalk':
            vec = rhs.contiguous().div(self.degree_mat.pow(0.5).view(-1, 1)) if self.transposed else rhs.contiguous()*self.degree_mat.pow(0.5).view(-1, 1)
        else:
            vec = rhs.contiguous()

        out = vec * self.laplacian_diag.view(-1, 1)
        out -= spmm(self.idx, self.laplacian_triu, self.operator_dimension, self.operator_dimension, vec)
        out -= spmm(torch.stack((self.idx[1], self.idx[0]), dim=0), self.laplacian_triu, self.operator_dimension, self.operator_dimension, vec)

        if self.normalization == "randomwalk":
            out *= self.degree_mat.pow(0.5).view(-1, 1) if self.transposed else self.degree_mat.pow(-0.5).view(-1, 1)

        return out

    def _size(self):
        return torch.Size([self.operator_dimension, self.operator_dimension])

    def _transpose_nonbatch(self):
        return GraphLaplacianOperator(self.x, self.idx, self.operator_dimension, self.graphbandwidth, self.normalization, self.self_loops, True) if self.normalization == 'randomwalk' else self

    def diagonalization(self: LinearOperator, method: str | None = None, num_modes: int = None) -> Tuple[Tensor | LinearOperator | None]:
        with gpytorch.settings.max_root_decomposition_size(3*num_modes if num_modes is not None and 3*num_modes <= self.shape[0] else self.shape[0]):
            if self.normalization == 'symmetric':
                evals, evecs = super().diagonalization(method)
                evals[0] = 0.0
                if num_modes is not None and num_modes < self.shape[0]:
                    evals, evecs = evals[:num_modes], evecs[:, :num_modes]
                return evals, evecs.to_dense()
            else:
                evals, evecs = GraphLaplacianOperator(self.x, self.idx, self.operator_dimension, self.graphbandwidth, 'symmetric', self.self_loops).diagonalization(method, num_modes)
                evecs *= self.degree_mat.pow(-0.5).view(-1, 1)
                evecs = normalize(evecs, p=2, dim=0)
                return evals, evecs

    def out_of_sample(self, x, edge_value, edge_idx):
        out = edge_value.div(-4*self.graphbandwidth.square()).exp()
        degree_test = out.sum(dim=1)
        out /= self.degree_unnorm_mat[edge_idx]*degree_test.view(-1, 1)

        if self.normalization == 'symmetric':
            out /= self.degree_mat.sqrt()[edge_idx]*out.sum(dim=1).sqrt().view(-1, 1)
        elif self.normalization == 'randomwalk':
            out /= out.sum(dim=1).view(-1, 1)

        out = out.unsqueeze(-1).mul(x[edge_idx]).sum(dim=1)
        return out
