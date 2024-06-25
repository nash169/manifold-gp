# !/usr/bin/env python
# encoding: utf-8

import torch


def graph_laplacian(edge_index, edge_value, graphbandwidth, dimension, normalization="unnormalized", self_loops=True):
    adjacency_unorm = torch.sparse_coo_tensor(edge_index, edge_value.div(-4*graphbandwidth.square()).exp().squeeze(), (dimension, dimension)).to_dense()
    adjacency_unorm = adjacency_unorm + adjacency_unorm.T
    if self_loops:
        adjacency_unorm += torch.eye(dimension).to(edge_value.device)
    degree_unorm = adjacency_unorm.sum(dim=1)

    adjacency = torch.mm(degree_unorm.pow(-1).diag(), torch.mm(adjacency_unorm, degree_unorm.pow(-1).diag()))
    degree = adjacency.sum(dim=1)

    if normalization == "symmetric":
        laplacian = (torch.eye(dimension).to(edge_value.device)-torch.mm(degree.pow(-0.5).diag(), torch.mm(adjacency, degree.pow(-0.5).diag()))) / graphbandwidth.square()
    elif normalization == "randomwalk":
        laplacian = (torch.eye(dimension).to(edge_value.device)-torch.mm(degree.pow(-1.0).diag(), adjacency)) / graphbandwidth.square()
    else:
        laplacian = (degree.diag() - adjacency) / graphbandwidth.square()

    return laplacian, adjacency_unorm, degree_unorm, adjacency, degree


def matern_precision(graph_laplacian, nu, lengthscale, degree_matrix=None):
    precision = torch.matrix_power(torch.eye(graph_laplacian.shape[0], device=graph_laplacian.device)*2*nu/lengthscale.square().squeeze() + graph_laplacian, nu)

    if degree_matrix is not None:
        precision = torch.mm(degree_matrix.diag(), precision)

    return precision


def matern_labeled_precision(matern_precision, mask):
    precision_xx = matern_precision[:, mask]
    precision_xx = precision_xx[mask, :]

    precision_xz = matern_precision[:, ~mask]
    precision_xz = precision_xz[mask, :]

    precision_zz = matern_precision[:, ~mask]
    precision_zz = precision_zz[~mask, :]

    precision_zx = matern_precision[:, mask]
    precision_zx = precision_zx[~mask, :]

    return precision_xx - torch.mm(precision_xz, torch.linalg.solve(precision_zz, precision_zx))


def matern_scaled_precision(matern_precision, outputscale):
    return matern_precision / outputscale


def matern_noisy_precision(matern_precision, noise_variance):
    return matern_precision - noise_variance*torch.mm(matern_precision, matern_precision) + noise_variance.square()*torch.mm(matern_precision, torch.mm(matern_precision, matern_precision))
