#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import trimesh
import networkx as nx

from mayavi import mlab


# Build ground truth on the mesh
def build_ground_truth(mesh_file):
    # Load mesh
    mesh = trimesh.load_mesh(trimesh.interfaces.gmsh.load_gmsh(mesh_file))

    # edges without duplication
    edges = mesh.edges_unique

    # the actual length of each unique edge
    length = mesh.edges_unique_length

    # create the graph with edge attributes for length
    graph = nx.Graph()
    for edge, L in zip(edges, length):
        graph.add_edge(*edge, length=L)

    # graph = trimesh.graph.vertex_adjacency_graph(mesh)

    geodesics = nx.shortest_path_length(graph, source=0, weight='length')

    N = len(mesh.vertices)

    ground_truth = np.zeros((N))
    period = 2  # 2*np.pi / 0.3 * 2

    for i in range(N):
        ground_truth[i] = 2 * np.sin(geodesics.get(i) * period + 0.3)

    return mesh.vertices, mesh.faces, ground_truth


# Plot function on the mesh
def plot_function(x, y, z, triangles, function):

    mlab.triangular_mesh(x, y, z, triangles, scalars=function)
    v_options = {'mode': 'sphere',
                 'scale_factor': 1e-2, }
    mlab.points3d(x[0], y[0], z[0], **v_options)


# RBF kernel
def squared_exp(x, y, sigma=1):
    l = -.5 / sigma**2
    xx = torch.einsum('ij,ij->i', x, x).unsqueeze(1)
    yy = torch.einsum('ij,ij->i', y, y).unsqueeze(0)
    k = -2 * torch.mm(x, y.T) + xx + yy
    k *= l
    return torch.exp(k)


# Edge probabilities
def edge_probability(x, y, t=1):
    xx = torch.einsum('ij,ij->i', x, x).unsqueeze(1)
    yy = torch.einsum('ij,ij->i', y, y).unsqueeze(0)
    k = -2 * torch.mm(x, y.T) + xx + yy
    return torch.exp(t*k)


def lanczos(A, v, iter=None):
    # Set number of iterations
    if iter is None:
        iter = v.size(0)

    # Create matrix for the Hamiltonian in Krylov subspace
    T = torch.zeros((iter, iter)).to(A.device)
    V = torch.zeros((v.size(0), iter)).to(A.device)

    # First step
    w = torch.mm(A, v)
    alpha = torch.mm(w.t(), v)
    w = w - alpha * v

    # Store
    T[0, 0] = alpha
    V[:, 0] = v.squeeze()

    for j in torch.arange(1, iter):
        beta = torch.linalg.norm(w)
        v = w/beta
        w = torch.mm(A, v)
        alpha = torch.mm(w.t(), v)
        w = w - alpha * v - beta*V[:, j-1].unsqueeze(1)

        T[j, j] = alpha
        T[j-1, j] = beta
        T[j, j-1] = beta
        V[:, j] = v.squeeze()

    return T, V
