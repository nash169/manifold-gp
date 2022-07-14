#!/usr/bin/env python
# encoding: utf-8

from src.utils import build_ground_truth, plot_function

mesh = 'rsc/torus.msh'

nodes, faces, truth = build_ground_truth(mesh)
x, y, z = (nodes[:, i] for i in range(3))
plot_function(x, y, z, faces, truth)
