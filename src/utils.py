#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import networkx as nx
import os

from mayavi import mlab


# Build ground truth on the mesh
def groundtruth_from_mesh(mesh_file):
    import trimesh
    # Load mesh
    if os.path.splitext(mesh_file)[1] == '.msh':
        mesh = trimesh.load_mesh(
            trimesh.interfaces.gmsh.load_gmsh(mesh_file))
    else:
        mesh = trimesh.load(mesh_file)

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
    period = 1  # 2*np.pi / 0.3 * 2

    for i in range(N):
        ground_truth[i] = 2 * np.sin(geodesics.get(i) * period + 0.3)

    return mesh.vertices, mesh.faces, ground_truth


def groundtruth_from_samples(X):
    import faiss
    index = faiss.IndexFlatL2(X.shape[1])
    index.train(X)
    index.add(X)
    distances, neighbors = index.search(X, 3)

    graph = nx.Graph()
    for i in range(X.shape[0]):
        graph.add_node(i, pos=(X[i, 0], X[i, 1]))
    for i in range(X.shape[0]):
        graph.add_edge(i, neighbors[i, 1], length=distances[i, 1])
        graph.add_edge(i, neighbors[i, 2], length=distances[i, 2])

    geodesics = nx.shortest_path_length(graph, source=0, weight='length')
    Y = np.zeros((X.shape[0]))
    for i in range(X.shape[0]):
        Y[i] = 0.5*np.sin(5e2 * geodesics.get(i)**2)

    return Y


def reduce_mesh(mesh_file):
    import trimesh

    mesh = trimesh.load_mesh(
        trimesh.interfaces.gmsh.load_gmsh("rsc/dragon.msh"))

    mesh = mesh.simplify_quadratic_decimation(10000)

    mesh.export('rsc/dragon_red.msh')


def load_mesh(mesh_file):
    import trimesh
    # Load mesh
    if os.path.splitext(mesh_file)[1] == '.msh':
        mesh = trimesh.load_mesh(trimesh.interfaces.gmsh.load_gmsh(mesh_file))
    else:
        mesh = trimesh.load(mesh_file)

    return mesh.vertices, mesh.faces


# Plot function on the mesh
def plot_function(x, y, z, triangles, function):
    mlab.figure()
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
