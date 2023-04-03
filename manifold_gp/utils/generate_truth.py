#!/usr/bin/env python
# encoding: utf-8

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


def groundtruth_from_samples(X):
    import faiss
    import networkx as nx
    import numpy as np

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
