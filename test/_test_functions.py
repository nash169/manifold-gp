# !/usr/bin/env python
# encoding: utf-8

import math
import torch
import gpytorch

from manifold_gp.utils.iostream import bcolors


def test_mv(dense_operator, sparse_operator, v):
    print("Test matrix-vector multiplication")
    dense_mv = torch.mm(dense_operator, v.view(-1, 1)).squeeze()[:10]
    sparse_mv = sparse_operator.matmul(v.view(-1, 1)).squeeze()[:10]
    print([round(elem, 5) for elem in dense_mv.tolist()])
    print([round(elem, 5) for elem in sparse_mv.tolist()])
    if [round(elem, 5) for elem in dense_mv.tolist()] == [round(elem, 5) for elem in sparse_mv.tolist()]:
        print(bcolors.OKGREEN + "Matrix-Vector Test: SUCCESS" + bcolors.ENDC)
    else:
        print(bcolors.FAIL + "Matrix-Vector Test: FAILED" + bcolors.ENDC)


def test_mv_transpose(dense_operator, sparse_operator, v):
    print("Test matrix(transpose)-vector multiplication")
    denset_mv = torch.mm(dense_operator.T, v.view(-1, 1)).squeeze()[:10]
    sparset_mv = sparse_operator.T.matmul(v.view(-1, 1)).squeeze()[:10]
    print([round(elem, 5) for elem in denset_mv.tolist()])
    print([round(elem, 5) for elem in sparset_mv.tolist()])
    if [round(elem, 5) for elem in denset_mv.tolist()] == [round(elem, 5) for elem in sparset_mv.tolist()]:
        print(bcolors.OKGREEN + "Matrix-Vector (transpose) Test: SUCCESS" + bcolors.ENDC)
    else:
        print(bcolors.FAIL + "Matrix-Vector (transpose) Test: FAILED" + bcolors.ENDC)


def test_diag(dense_operator, sparse_operator):
    print("Test matrix(transpose)-vector multiplication")
    dense_diag = dense_operator.diag()[:10]
    sparse_diag = sparse_operator.diagonal()[:10]
    print([round(elem, 5) for elem in dense_diag.tolist()])
    print([round(elem, 5) for elem in sparse_diag.tolist()])
    if [round(elem, 5) for elem in dense_diag.tolist()] == [round(elem, 5) for elem in sparse_diag.tolist()]:
        print(bcolors.OKGREEN + "Diagonal Test: SUCCESS" + bcolors.ENDC)
    else:
        print(bcolors.FAIL + "Diagonal Test: FAILED" + bcolors.ENDC)


def test_solve(dense_operator, sparse_operator, v):
    print("Test matrix-vector solve")
    dense_solve = torch.linalg.solve(dense_operator, v)[:10]
    sparse_solve = sparse_operator.solve(v.view(-1, 1)).squeeze()[:10]
    print([round(elem, 5) for elem in dense_solve.tolist()])
    print([round(elem, 5) for elem in sparse_solve.tolist()])
    if [round(elem, 5) for elem in dense_solve.tolist()] == [round(elem, 5) for elem in sparse_solve.tolist()]:
        print(bcolors.OKGREEN + "Solve Test: SUCCESS" + bcolors.ENDC)
    else:
        print(bcolors.FAIL + "Solve Test: FAILED" + bcolors.ENDC)


def test_grad(dense_operator, sparse_operator, v, *params):
    print("Test gradient")
    loss = torch.mm(dense_operator.T, v.view(-1, 1)).sum()
    loss.backward()
    dense_grad = [value.grad.item() for value in params]
    [value.grad.data.zero_() for value in params]
    loss = sparse_operator.T.matmul(v.view(-1, 1)).sum()
    loss.backward()
    sparse_grad = [value.grad.item() for value in params]
    [value.grad.data.zero_() for value in params]
    print([round(elem, 5) for elem in dense_grad])
    print([round(elem, 5) for elem in sparse_grad])
    if [round(elem, 5) for elem in dense_grad] == [round(elem, 5) for elem in sparse_grad]:
        print(bcolors.OKGREEN + "Gradient Test: SUCCESS" + bcolors.ENDC)
    else:
        print(bcolors.FAIL + "Gradient Test: FAILED" + bcolors.ENDC)


def test_ml(dense_operator, sparse_operator, v, max_cholesky=800, cg_tolerance=1e-2, cg_max_iter=1000, **params):
    loss = 0.5 * sum([torch.dot(v, torch.mv(dense_operator, v)), -torch.logdet(dense_operator), v.size(-1) * math.log(2 * math.pi)])
    loss.backward()
    loss_dense = loss.item()
    dense_grad = [value.grad.item() for _, value in params.items()]
    [value.grad.data.zero_() for _, value in params.items()]
    with gpytorch.settings.max_cholesky_size(max_cholesky), gpytorch.settings.cg_tolerance(cg_tolerance),  gpytorch.settings.max_cg_iterations(cg_max_iter):
        loss = 0.5 * sum([torch.dot(v, sparse_operator.matmul(v.view(-1, 1)).squeeze()),
                          -sparse_operator.inv_quad_logdet(logdet=True)[1],
                          v.shape[0] * math.log(2 * math.pi)])
    loss.backward()
    loss_sparse = loss.item()
    sparse_grad = [value.grad.item() for _, value in params.items()]
    [value.grad.data.zero_() for _, value in params.items()]
    print("Test Loss")
    print(loss_dense)
    print(loss_sparse)
    if loss_dense == loss_sparse:
        print(bcolors.OKGREEN + "Gradient Test: SUCCESS" + bcolors.ENDC)
    else:
        print(bcolors.FAIL + "Gradient Test: FAILED" + bcolors.ENDC)
    print("Test Gradient Loss")
    print([round(elem, 5) for elem in dense_grad])
    print([round(elem, 5) for elem in sparse_grad])
    if [round(elem, 5) for elem in dense_grad] == [round(elem, 5) for elem in sparse_grad]:
        print(bcolors.OKGREEN + "Gradient Test: SUCCESS" + bcolors.ENDC)
    else:
        print(bcolors.FAIL + "Gradient Test: FAILED" + bcolors.ENDC)


def test_eigen(dense_operator, sparse_operator, normalization, max_cholesky=800, cg_tolerance=1e-2, cg_max_iter=1000):
    print("Test Eigen")
    with torch.no_grad():
        if normalization == 'randomwalk':
            dense_eval, dense_evec = torch.linalg.eig(dense_operator)
            dense_eval, dense_evec = torch.real(dense_eval), torch.real(dense_evec)
            dense_eval, dense_eval_idx = torch.sort(dense_eval)
            dense_evec = dense_evec[:, dense_eval_idx]
        else:
            dense_eval, dense_evec = torch.linalg.eigh(dense_operator)
        with gpytorch.settings.max_cholesky_size(max_cholesky), gpytorch.settings.cg_tolerance(cg_tolerance),  gpytorch.settings.max_cg_iterations(cg_max_iter):
            sparse_eval, sparse_evec = sparse_operator.diagonalization()
    print([round(elem, 5) for elem in dense_eval[:10].tolist()])
    print([round(elem, 5) for elem in sparse_eval[:10].tolist()])
    if [round(elem, 5) for elem in dense_eval[1:10].tolist()] == [round(elem, 5) for elem in sparse_eval[1:10].tolist()]:
        print(bcolors.OKGREEN + "Eigenvalues Test: SUCCESS" + bcolors.ENDC)
    else:
        print(bcolors.FAIL + "Eigenvalues Test: FAILED" + bcolors.ENDC)
    print([round(elem, 5) for elem in dense_evec[:10, 1].tolist()])
    print([round(elem, 5) for elem in sparse_evec[:10, 1].tolist()])
    if [round(elem, 5) for elem in dense_evec[:10, 1].tolist()] == [round(elem, 5) for elem in sparse_evec[:10, 1].tolist()]:
        print(bcolors.OKGREEN + "Eigenvectors Test: SUCCESS" + bcolors.ENDC)
    else:
        print(bcolors.FAIL + "Eigenvectors Test: FAILED" + bcolors.ENDC)
    return dense_eval, dense_evec, sparse_eval, sparse_evec


def test_outofsample(edge_value, edge_index, dense_eval, dense_evec, sparse_operator, sparse_eval, sparse_evec, dense_degree_unorm, dense_degree, graphbandwidth, lengthscale, nu, normalization):
    rows, cols = torch.arange(edge_index.shape[0], device=edge_index.device).repeat_interleave(edge_index.shape[1]), edge_index.reshape(1, -1).squeeze()
    adjacency_ext_unorm = torch.sparse_coo_tensor(torch.stack([rows, cols], dim=0), edge_value.reshape(
        1, -1).squeeze().div(-4*graphbandwidth.square()).exp().squeeze(), (edge_index.shape[0], dense_evec.shape[0])).to_dense()
    degree_ext_unorm = adjacency_ext_unorm.sum(dim=1)
    adjacency_ext = torch.mm(degree_ext_unorm.pow(-1).diag(), torch.mm(adjacency_ext_unorm, dense_degree_unorm.pow(-1).diag()))
    degree_ext = adjacency_ext.sum(dim=1)
    if normalization == 'symmetric':
        ext_mat = torch.mm(degree_ext.pow(-0.5).diag(), torch.mm(adjacency_ext, dense_degree.pow(-0.5).diag()))
    elif normalization == 'randomwalk':
        ext_mat = torch.mm(degree_ext.pow(-1.0).diag(), adjacency_ext)

    dense_spectral_density = (2*nu / lengthscale.square() + dense_eval).pow(-nu)
    dense_spectral_density /= (1 - dense_eval*graphbandwidth.square()).square()
    dense_spectral_density /= dense_spectral_density.sum()
    dense_spectral_density *= dense_evec.shape[0]
    dense_evec_ext = dense_spectral_density.sqrt()*torch.mm(ext_mat, dense_evec)

    sparse_spectral_density = (2*nu / lengthscale.square() + sparse_eval).pow(-nu)
    sparse_spectral_density /= (1 - sparse_eval*graphbandwidth.square()).square()
    sparse_spectral_density /= sparse_spectral_density.sum()
    sparse_spectral_density *= sparse_evec.shape[0]
    sparse_evec_ext = sparse_spectral_density.sqrt()*sparse_operator.out_of_sample(sparse_evec, edge_value, edge_index)

    print([round(elem, 5) for elem in dense_evec_ext[:10, 1].tolist()])
    print([round(elem, 5) for elem in sparse_evec_ext[:10, 1].tolist()])
    if [round(elem, 5) for elem in dense_evec_ext[:10, 1].tolist()] == [round(elem, 5) for elem in sparse_evec_ext[:10, 1].tolist()]:
        print(bcolors.OKGREEN + "Out of Sample Test: SUCCESS" + bcolors.ENDC)
    else:
        print(bcolors.FAIL + "Out of Sample Test: FAILED" + bcolors.ENDC)
