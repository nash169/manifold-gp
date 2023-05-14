#!/usr/bin/env python
# encoding: utf-8

import torch
from typing import Tuple
from linear_operator import LinearOperator


class LaplacianRandomWalkOperator(LinearOperator):
    def __init__(self, values, degree, epsilon, indices, size, kernel):
        super(LaplacianRandomWalkOperator, self).__init__(values, degree, epsilon, indices=indices, size=size, kernel=kernel)

    def _matmul(self, x):
        return x.index_add(0, self._kwargs['indices'][0, :], self._args[0].view(-1, 1) * x[self._kwargs['indices'][1, :]], alpha=-1).div(self._args[2].square())

    def _matmul_block(self, rhs, rows, cols):
        mask = torch.isin(self._kwargs['indices'][0], rows)*torch.isin(self._kwargs['indices'][1], cols)
        idx_rows = torch.where(self._kwargs['indices'][0][mask].unsqueeze(-1) == rows)[1]
        idx_cols = torch.where(self._kwargs['indices'][1][mask].unsqueeze(-1) == cols)[1]

        if torch.equal(rows, cols):
            return rhs.index_add(0, idx_rows, self._args[0][mask].view(-1, 1) * rhs[idx_cols], alpha=-1).div(self._args[2].square())
        else:
            return torch.zeros(rows.shape[0], rhs.shape[1]).index_add(0, idx_rows, self._args[0][mask].view(-1, 1) * rhs[idx_cols], alpha=-1).div(self._args[2].square())

    def _size(self):
        return torch.Size([self._kwargs['size'], self._kwargs['size']])

    def _transpose_nonbatch(self):
        return self

    def diagonalization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # evals, evecs = self._kwargs['kernel'].laplacian(operator='symmetric').diagonalization()
        # evecs = evecs.mul(self._args[1].sqrt().view(-1, 1))

        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import eigsh

        # params
        epsilon = self._kwargs['kernel'].epsilon[0]
        num_points = self._kwargs['size']
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
