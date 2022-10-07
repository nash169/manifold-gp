#!/usr/bin/env python
# encoding: utf-8

import torch

from src.laplacian import Laplacian
from src.utils import build_ground_truth

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load mesh
mesh = 'rsc/torus3k.msh'
nodes, faces, truth = build_ground_truth(mesh)
x, y, z = (nodes[:, i] for i in range(3))
X_sampled = torch.from_numpy(nodes).float().to(device).requires_grad_(True)
Y_sampled = torch.from_numpy(truth).float().to(device).requires_grad_(True)

lp = Laplacian()
lp.samples = X_sampled
print(lp.eps_)
lp.train(Y_sampled.unsqueeze(1))
print(lp.eps_)
