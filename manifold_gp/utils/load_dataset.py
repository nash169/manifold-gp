#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np
from importlib.resources import files


def manifold_1D_dataset():
    data_path = files('manifold_gp.data').joinpath('dumbbell.msh')
    data = get_data(data_path, "Nodes", "Elements")

    vertices = data['Nodes'][:, 1:-1]
    edges = data['Elements'][:, -2:].astype(int) - 1
    truth, _ = groundtruth_from_samples(vertices, edges)

    return torch.from_numpy(vertices).float(), torch.from_numpy(truth).float(), edges


def manifold_2D_dataset():
    data_path = files('manifold_gp.data').joinpath('dragon.stl')
    nodes, _, truth = groundtruth_from_mesh(data_path)

    return torch.from_numpy(nodes).float(), torch.from_numpy(truth).float()


def rmnist_dataset(normalized=True):
    import os

    if os.path.isfile('manifold_gp/data/rmnist_train_x.npy'):
        sampled_x, sampled_y = np.load('manifold_gp/data/rmnist_train_x.npy'), np.load('manifold_gp/data/rmnist_train_y.npy')
        test_x, test_y = np.load('manifold_gp/data/rmnist_test_x.npy'), np.load('manifold_gp/data/rmnist_test_y.npy')
    else:
        import tensorflow as tf
        from .rotate_mnist import rotate_mnist
        (train_samples, train_labels), (test_samples, test_labels) = tf.keras.datasets.mnist.load_data()
        sampled_x, sampled_y, _ = rotate_mnist(train_samples, train_labels, num_samples=100, rots_sample=10)
        test_x, test_y, _ = rotate_mnist(test_samples, test_labels, num_samples=10, rots_sample=10)

    if normalized:
        sampled_x = (sampled_x - (255. / 2.0)) / 255.
        test_x = (test_x - (255. / 2.0)) / 255.

    return torch.from_numpy(sampled_x).float(), torch.from_numpy(sampled_y).float(), torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float()


def groundtruth_from_samples(vertices, edges):
    import networkx as nx
    import numpy as np

    graph = nx.Graph()

    # add nodes
    for i in range(vertices.shape[0]):
        graph.add_node(i, pos=(vertices[i, 0], vertices[i, 1]))

    # edges
    for i in range(edges.shape[0]):
        graph.add_edge(edges[i, 0], edges[i, 1], length=np.linalg.norm(vertices[edges[i, 0], :]-vertices[edges[i, 1], :]))

    # geodesics
    geodesics = nx.shortest_path_length(graph, source=0, weight='length')

    # function
    period = 1.5
    phase = 0.0
    truth = np.zeros((vertices.shape[0]))
    for i in range(truth.shape[0]):
        truth[i] = 2 * np.sin(geodesics.get(i) * period + phase)

    return truth, geodesics


def groundtruth_from_mesh(mesh_file):
    import os
    import numpy as np
    import trimesh
    import networkx as nx

    # Load mesh
    if os.path.splitext(mesh_file)[1] == '.msh':
        mesh = trimesh.load_mesh(
            trimesh.interfaces.gmsh.load_gmsh(str(mesh_file)))
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


def get_data(file_path, *args):
    M = {}
    for var in args:
        with open(file_path) as fs:
            g = _get_line(var, fs)
            M[var] = np.loadtxt(g)

    return M


def _get_line(name, fs):
    while True:
        try:
            line = next(fs)
        except StopIteration:
            return

        if name in line:
            try:
                next(fs)
            except StopIteration:
                return

            while True:
                try:
                    line = next(fs)
                except StopIteration:
                    return

                if line in ["\n", "\r\n"]:
                    break
                else:
                    yield line
        elif not line:
            break
