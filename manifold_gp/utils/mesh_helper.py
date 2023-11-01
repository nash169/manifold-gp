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


# def groundtruth_from_samples(X):
#     import faiss
#     import networkx as nx
#     import numpy as np

#     index = faiss.IndexFlatL2(X.shape[1])
#     index.train(X)
#     index.add(X)
#     distances, neighbors = index.search(X, 3)

#     graph = nx.Graph()
#     for i in range(X.shape[0]):
#         graph.add_node(i, pos=(X[i, 0], X[i, 1]))
#     for i in range(X.shape[0]):
#         graph.add_edge(i, neighbors[i, 1], length=distances[i, 1])
#         graph.add_edge(i, neighbors[i, 2], length=distances[i, 2])

#     geodesics = nx.shortest_path_length(graph, source=0, weight='length')
#     Y = np.zeros((X.shape[0]))
#     period = 2*np.pi / 0.3 * 2
#     for i in range(X.shape[0]):
#         Y[i] = 2 * np.sin(np.power(geodesics.get(i), 2) * period + 0.3)
#     return Y

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


def reduce_mesh(mesh_file):
    import trimesh

    mesh = trimesh.load_mesh(
        trimesh.interfaces.gmsh.load_gmsh(mesh_file))

    mesh = mesh.simplify_quadratic_decimation(10000)

    mesh.export('rsc/dragon_red.msh')


def load_mesh(mesh_file):
    import os
    import trimesh

    # Load mesh
    if os.path.splitext(mesh_file)[1] == '.msh':
        mesh = trimesh.load_mesh(trimesh.interfaces.gmsh.load_gmsh(mesh_file))
    else:
        mesh = trimesh.load(mesh_file)

    return mesh.vertices, mesh.faces


def plot_1D(fig, ax, vertices, edges, gradient):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    vertices = vertices.reshape(-1, 1, 2)
    segments = np.concatenate([vertices[edges[:, 0]], vertices[edges[:, 1]]], axis=1)

    norm = plt.Normalize(gradient.min(), gradient.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(gradient)
    lc.set_linewidth(5)
    line = ax.add_collection(lc)

    fig.colorbar(line, ax=ax)
    ax.set_xlim(vertices[:, 0, 0].min(), vertices[:, 0, 0].max())
    ax.set_ylim(vertices[:, 0, 1].min(), vertices[:, 0, 1].max())

    ax.axis('equal')
