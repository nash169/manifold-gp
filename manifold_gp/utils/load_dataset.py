#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np
from importlib.resources import files
import os


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


def rmnist_dataset(scaling=True, single_digit=False, regenerate=False):
    if single_digit:
        if os.path.isfile(files('manifold_gp.data').joinpath('srmnist_train_x.npy')) and not regenerate:
            print("Loading SRMNIST")
            sampled_x, sampled_y, sampled_labels = np.load(files('manifold_gp.data').joinpath('srmnist_train_x.npy')), np.load(
                files('manifold_gp.data').joinpath('srmnist_train_y.npy')), np.load(files('manifold_gp.data').joinpath('srmnist_train_labels.npy'))
            test_x, test_y, test_labels = np.load(files('manifold_gp.data').joinpath('srmnist_test_x.npy')), np.load(
                files('manifold_gp.data').joinpath('srmnist_test_y.npy')), np.load(files('manifold_gp.data').joinpath('srmnist_test_labels.npy'))
        else:
            print("Generating SRMNIST")
            import tensorflow as tf
            from .rotate_mnist import rotate_mnist
            # generate
            (train_samples, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
            digits_idx = [1, 8, 5, 7, 2, 0, 18, 15, 17, 4]
            sampled_x, sampled_y, sampled_labels = rotate_mnist(train_samples[digits_idx], train_labels[digits_idx], num_samples=len(digits_idx), rots_sample=1000)
            test_x, test_y, test_labels = rotate_mnist(train_samples[digits_idx], train_labels[digits_idx], num_samples=len(digits_idx), rots_sample=100)
            # # shuffle
            # rand_idx = np.random.permutation(sampled_x.shape[0])
            # sampled_x, sampled_y, sampled_labels = sampled_x[rand_idx], sampled_y[rand_idx], sampled_labels[rand_idx]
            # rand_idx = np.random.permutation(test_x.shape[0])
            # test_x, test_y, test_labels = test_x[rand_idx], test_y[rand_idx], test_labels[rand_idx]
            # save
            np.save(files('manifold_gp.data').joinpath('srmnist_train_x.npy'), sampled_x)
            np.save(files('manifold_gp.data').joinpath('srmnist_train_y.npy'), sampled_y)
            np.save(files('manifold_gp.data').joinpath('srmnist_train_labels.npy'), sampled_labels)
            np.save(files('manifold_gp.data').joinpath('srmnist_test_x.npy'), test_x)
            np.save(files('manifold_gp.data').joinpath('srmnist_test_y.npy'), test_y)
            np.save(files('manifold_gp.data').joinpath('srmnist_test_labels.npy'), test_labels)
    else:
        if os.path.isfile(files('manifold_gp.data').joinpath('rmnist_train_x.npy')) and not regenerate:
            print("Loading RMNIST")
            sampled_x, sampled_y, sampled_labels = np.load(files('manifold_gp.data').joinpath('rmnist_train_x.npy')), np.load(
                files('manifold_gp.data').joinpath('rmnist_train_y.npy')), np.load(files('manifold_gp.data').joinpath('rmnist_train_labels.npy'))
            test_x, test_y, test_labels = np.load(files('manifold_gp.data').joinpath('rmnist_test_x.npy')), np.load(
                files('manifold_gp.data').joinpath('rmnist_test_y.npy')), np.load(files('manifold_gp.data').joinpath('rmnist_test_labels.npy'))
        else:
            print("Generating RMNIST")
            import tensorflow as tf
            from .rotate_mnist import rotate_mnist
            # generate
            (train_samples, train_labels), (test_samples, test_labels) = tf.keras.datasets.mnist.load_data()
            sampled_x, sampled_y, sampled_labels = rotate_mnist(train_samples, train_labels, num_samples=100, rots_sample=100)
            test_x, test_y, test_labels = rotate_mnist(test_samples, test_labels, num_samples=100, rots_sample=10)
            # # shuffle
            # rand_idx = np.random.permutation(sampled_x.shape[0])
            # sampled_x, sampled_y, sampled_labels = sampled_x[rand_idx], sampled_y[rand_idx], sampled_labels[rand_idx]
            # rand_idx = np.random.permutation(test_x.shape[0])
            # test_x, test_y, test_labels = test_x[rand_idx], test_y[rand_idx], test_labels[rand_idx]
            # save
            np.save(files('manifold_gp.data').joinpath('rmnist_train_x.npy'), sampled_x)
            np.save(files('manifold_gp.data').joinpath('rmnist_train_y.npy'), sampled_y)
            np.save(files('manifold_gp.data').joinpath('rmnist_train_labels.npy'), sampled_labels)
            np.save(files('manifold_gp.data').joinpath('rmnist_test_x.npy'), test_x)
            np.save(files('manifold_gp.data').joinpath('rmnist_test_y.npy'), test_y)
            np.save(files('manifold_gp.data').joinpath('rmnist_test_labels.npy'), test_labels)

    if scaling:
        sampled_x = (sampled_x - (255. / 2.0)) / 255.
        test_x = (test_x - (255. / 2.0)) / 255.

    return torch.from_numpy(sampled_x).float(), torch.from_numpy(sampled_y).float(), torch.from_numpy(sampled_labels).int(), torch.from_numpy(test_x).float(), torch.from_numpy(test_y).float(), torch.from_numpy(test_labels).int()


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
