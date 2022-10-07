#!/usr/bin/env python
# encoding: utf-8

import trimesh

mesh = trimesh.load_mesh(trimesh.interfaces.gmsh.load_gmsh("rsc/dragon.msh"))

mesh = mesh.simplify_quadratic_decimation(10000)

mesh.export('rsc/dragon_red.msh')
