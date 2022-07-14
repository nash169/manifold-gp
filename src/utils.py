#!/usr/bin/env python
# encoding: utf-8

import numpy as np

# Build ground truth on the mesh


def build_ground_truth(mesh_file):
    import trimesh
    import networkx as nx

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
    from mayavi import mlab
    mlab.triangular_mesh(x, y, z, triangles, scalars=function)
    v_options = {'mode': 'sphere',
                 'scale_factor': 1e-2, }
    mlab.points3d(x[0], y[0], z[0], **v_options)

# RBF kernel


def squared_exp(x, y, sigma=1, eta=1):
    l = -.5 / sigma**2
    xx = torch.einsum('ij,ij->i', x, x).unsqueeze(1)
    yy = torch.einsum('ij,ij->i', y, y).unsqueeze(0)
    k = -2 * torch.mm(x, y.T) + xx + yy
    k *= l
    return eta*torch.exp(k)
