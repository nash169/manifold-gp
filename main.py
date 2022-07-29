#!/usr/bin/env python
# encoding: utf-8

import faiss
import torch
import numpy as np
from torch_geometric.nn import knn_graph
from src.utils import build_ground_truth, plot_function, edge_probability

# CPU/GPU setting
use_cuda = torch.cuda.is_available()
device = "cpu"  # torch.device("cuda" if use_cuda else "cpu")

# Load mesh
mesh = 'rsc/torus.msh'
nodes, faces, truth = build_ground_truth(mesh)
# x, y, z = (nodes[:, i] for i in range(3))
# plot_function(x, y, z, faces, truth)

k = 10
X = torch.from_numpy(np.array(nodes)).float().to(device).requires_grad_(True)
P = edge_probability(X, X)

# edge_index = knn_graph(X, k, None, loop=True)


# x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
# batch = torch.tensor([0, 0, 1, 1])
# edge_index = knn_graph(x, k=2, batch=batch, loop=False)
