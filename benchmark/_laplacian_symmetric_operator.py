#!/usr/bin/env python
# encoding: utf-8

from typing import Union

import torch

from jaxtyping import Float, Int32
from torch import Tensor

from linear_operator.operators._linear_operator import LinearOperator
from linear_operator.utils.memoize import cached

from torch_sparse import spmm


class LaplacianSymmetricOperator(LinearOperator):
    def __init__(
            self,
            x: Float[Tensor, "N"],
            idx: Int32[Tensor, "N"],
            operator_dimension: int,
            graphbandwidth: Tensor,
    ):
        super().__init__(
            x,
            idx=idx,
            operator_dimension=operator_dimension,
            graphbandwidth=graphbandwidth,
        )
        self.x = x
        self.idx = idx
        self.operator_dimension = operator_dimension
        self.graphbandwidth = graphbandwidth

    @property
    @cached(name="adjacency_unnorm_mat")
    def adjacency_unnorm_mat(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "M"]:
        # print("adjacency_unnorm_mat")
        return self.x.div(-4*self.graphbandwidth.square()).exp().squeeze()

    @property
    @cached(name="degree_unnorm_mat")
    def degree_unnorm_mat(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "N"]:
        # print("degree_unnorm_mat")
        return torch.ones(self.operator_dimension) \
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
        return self.degree_unnorm_mat.pow(-2) \
            .scatter_add_(0, self.idx[0, :], self.adjacency_mat) \
            .scatter_add(0, self.idx[1, :], self.adjacency_mat)

    @property
    @cached(name="laplacian_diag")
    def _diagonal(self: LinearOperator) -> Tensor:
        # print("laplacian_diag")
        return (1 - self.degree_unnorm_mat.pow(-2)*self.degree_mat.pow(-1))

    @property
    @cached(name="laplacian_mat")
    def laplacian_mat(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "M"]:
        degree_sqrt = self.degree_mat.sqrt()
        return self.adjacency_mat.div(degree_sqrt[self.idx[0, :]]*degree_sqrt[self.idx[1, :]])

    # def _matmul(
    #     self: Float[LinearOperator, "N N"],
    #     rhs: Union[Float[torch.Tensor, "N C"], Float[torch.Tensor, "N"]],
    # ):
    #     return (rhs.contiguous() * self._diagonal.view(-1, 1)) \
    #         .index_add(0, self.idx[0], self.laplacian_mat.view(-1, 1) * rhs.contiguous()[self.idx[1]], alpha=-1) \
    #         .index_add(0, self.idx[1], self.laplacian_mat.view(-1, 1) * rhs.contiguous()[self.idx[0]], alpha=-1).div(self.graphbandwidth.square())

    def _matmul(
        self: Float[LinearOperator, "N N"],
        rhs: Union[Float[torch.Tensor, "N C"], Float[torch.Tensor, "N"]],
    ):
        return (rhs.contiguous() * self._diagonal.view(-1, 1)
                - spmm(self.idx, self.laplacian_mat, self.operator_dimension, self.operator_dimension, rhs.contiguous())
                - spmm(torch.stack((self.idx[1], self.idx[0]), dim=0), self.laplacian_mat, self.operator_dimension, self.operator_dimension, rhs.contiguous())) \
            .div(self.graphbandwidth.square())

    # @property
    # @cached(name="laplacian_mat")
    # def laplacian_mat(self: Float[LinearOperator, "N N"]) -> torch.Tensor:
    #     print("laplacian_mat")
    #     degree_sqrt = self.degree_mat.sqrt()
    #     return torch.sparse_coo_tensor(self.idx, self.adjacency_mat.div(degree_sqrt[self.idx[0, :]]*degree_sqrt[self.idx[1, :]]), (self.operator_dimension, self.operator_dimension))

    # def _matmul(
    #     self: Float[LinearOperator, "N N"],
    #     rhs: Union[Float[torch.Tensor, "N C"], Float[torch.Tensor, "N"]],
    # ):
    #     return (rhs.contiguous() * self._diagonal.view(-1, 1)) - torch.sparse.mm(self.laplacian_mat, rhs) - torch.sparse.mm(self.laplacian_mat.T, rhs)

    def _size(self):
        return torch.Size([self.operator_dimension, self.operator_dimension])

    def _transpose_nonbatch(self):
        return self


class _LaplacianSymmetricOperator(LinearOperator):
    def __init__(
            self,
            x: Float[Tensor, "N"],
            idx: Int32[Tensor, "N"],
            operator_dimension: int,
            graphbandwidth: Tensor
    ):
        super().__init__(
            x,
            idx=idx,
            operator_dimension=operator_dimension,
            graphbandwidth=graphbandwidth,
        )
        self.x = x
        self.idx = idx
        self.operator_dimension = operator_dimension
        self.graphbandwidth = graphbandwidth

    # @property
    # @cached(name="adjacency_unnorm_mat")
    # def adjacency_unnorm_mat(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "M"]:
    #     return self.x.div(-4*self.graphbandwidth.square()).exp().squeeze()

    # @property
    # @cached(name="degree_unnorm_mat")
    # def degree_unnorm_mat(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "N"]:
    #     # return torch.zeros(self.operator_dimension).scatter_add_(0, self.idx[0, :], self.adjacency_unnorm_mat).scatter_add_(0, self.idx[1, :], self.adjacency_unnorm_mat)
    #     return self.adjacency_unnorm_mat[:self.operator_dimension] \
    #         .scatter_add(0, self.idx[0, self.operator_dimension:], self.adjacency_unnorm_mat[self.operator_dimension:]) \
    #         .scatter_add(0, self.idx[1, self.operator_dimension:], self.adjacency_unnorm_mat[self.operator_dimension:])

    # @property
    # @cached(name="adjacency_mat")
    # def adjacency_mat(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "M"]:
    #     # adjacency_unnorm_mat = self.x.div(-4*self.graphbandwidth.square()).exp().squeeze()
    #     # degree_unnorm_mat = torch.zeros(self.operator_dimension).scatter_add_(0, self.idx[0, :], adjacency_unnorm_mat).scatter_add_(0, self.idx[1, :], adjacency_unnorm_mat)
    #     return self.adjacency_unnorm_mat.div(self.degree_unnorm_mat[self.idx[0, :]]*self.degree_unnorm_mat[self.idx[1, :]])

    @property
    @cached(name="adjacency_mat")
    def adjacency_mat(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "M"]:
        adjacency_unnorm_mat = self.x.div(-4*self.graphbandwidth.square()).exp().squeeze()
        degree_unnorm_mat = adjacency_unnorm_mat[:self.operator_dimension] \
            .scatter_add(0, self.idx[0, self.operator_dimension:], adjacency_unnorm_mat[self.operator_dimension:]) \
            .scatter_add(0, self.idx[1, self.operator_dimension:], adjacency_unnorm_mat[self.operator_dimension:])
        return adjacency_unnorm_mat.div(degree_unnorm_mat[self.idx[0, :]]*degree_unnorm_mat[self.idx[1, :]])

    @property
    @cached(name="degree_mat")
    def degree_mat(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "N"]:
        # adjacency_mat = self.adjacency_mat
        # return torch.zeros(self.operator_dimension).scatter_add_(0, self.idx[0, :], self.adjacency_mat).scatter_add(0, self.idx[1, :], self.adjacency_mat)
        return self.adjacency_mat[:self.operator_dimension] \
            .scatter_add(0, self.idx[0, self.operator_dimension:], self.adjacency_mat[self.operator_dimension:]) \
            .scatter_add(0, self.idx[1, self.operator_dimension:], self.adjacency_mat[self.operator_dimension:])

    @property
    @cached(name="laplacian_mat")
    def laplacian_mat(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "M"]:
        degree_mat = self.degree_mat.sqrt()
        return self.adjacency_mat.div(degree_mat[self.idx[0, :]]*degree_mat[self.idx[1, :]])

    def _matmul(
        self: Float[LinearOperator, "N N"],
        rhs: Union[Float[torch.Tensor, "N C"], Float[torch.Tensor, "N"]],
    ):
        # return rhs.contiguous() \
        #     .index_add(0, self.idx[0, :], self.laplacian_mat.view(-1, 1) * rhs[self.idx[1, :]], alpha=-1) \
        #     .index_add(0, self.idx[1, :], self.laplacian_mat.view(-1, 1) * rhs[self.idx[0, :]], alpha=-1).div(self.graphbandwidth.square())
        return (rhs.contiguous()*(1-self.laplacian_mat[:self.operator_dimension].view(-1, 1))) \
            .index_add(0, self.idx[0, self.operator_dimension:], self.laplacian_mat[self.operator_dimension:].view(-1, 1) * rhs.contiguous()[self.idx[1, self.operator_dimension:]], alpha=-1) \
            .index_add(0, self.idx[1, self.operator_dimension:], self.laplacian_mat[self.operator_dimension:].view(-1, 1) * rhs.contiguous()[self.idx[0, self.operator_dimension:]], alpha=-1) \
            .div(self.graphbandwidth.square())

    def _size(self):
        return torch.Size([self.operator_dimension, self.operator_dimension])

    def _transpose_nonbatch(self):
        return self


class __LaplacianSymmetricOperator(LinearOperator):
    def __init__(
            self,
            x: Float[Tensor, "N"],
            idx: Int32[Tensor, "N"],
            operator_dimension: int,
            graphbandwidth: Tensor
    ):
        super().__init__(
            x,
            idx=idx,
            operator_dimension=operator_dimension,
            graphbandwidth=graphbandwidth,
        )
        self.x = x
        self.idx = idx
        self.operator_dimension = operator_dimension
        self.graphbandwidth = graphbandwidth

    @property
    @cached(name="adjacency_mat")
    def adjacency_mat(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "M"]:
        adjacency_unnorm_mat = self.x.div(-4*self.graphbandwidth.square()).exp().squeeze()
        degree_unnorm_mat = torch.zeros(self.operator_dimension).scatter_add_(0, self.idx[0, :], adjacency_unnorm_mat)
        return adjacency_unnorm_mat.div(degree_unnorm_mat[self.idx[0, :]]*degree_unnorm_mat[self.idx[1, :]])

    @property
    @cached(name="degree_mat")
    def degree_mat(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "N"]:
        return torch.zeros(self.operator_dimension).scatter_add_(0, self.idx[0, :], self.adjacency_mat)

    # @property
    # @cached(name="laplacian_mat")
    # def laplacian_mat(self: Float[LinearOperator, "N N"]) -> Float[torch.Tensor, "M"]:
    #     degree_mat = self.degree_mat.sqrt()
    #     return self.adjacency_mat.div(degree_mat[self.idx[0, :]]*degree_mat[self.idx[1, :]])

    # def _matmul(
    #     self: Float[LinearOperator, "N N"],
    #     rhs: Union[Float[torch.Tensor, "N C"], Float[torch.Tensor, "N"]],
    # ):
    #     return rhs.contiguous().index_add(0, self.idx[0, :], self.laplacian_mat.view(-1, 1) * rhs.contiguous()[self.idx[1, :]], alpha=-1).div(self.graphbandwidth.square())

    @property
    @cached(name="laplacian_mat")
    def laplacian_mat(self: Float[LinearOperator, "N N"]) -> torch.Tensor:
        degree_sqrt = self.degree_mat.sqrt()
        return torch.sparse_coo_tensor(self.idx, self.adjacency_mat.div(degree_sqrt[self.idx[0, :]]*degree_sqrt[self.idx[1, :]]), (self.operator_dimension, self.operator_dimension))

    def _matmul(
        self: Float[LinearOperator, "N N"],
        rhs: Union[Float[torch.Tensor, "N C"], Float[torch.Tensor, "N"]],
    ):
        return (rhs.contiguous()-torch.sparse.mm(self.laplacian_mat, rhs.contiguous())).div(self.graphbandwidth.square())

    def _size(self):
        return torch.Size([self.operator_dimension, self.operator_dimension])

    def _transpose_nonbatch(self):
        return self
