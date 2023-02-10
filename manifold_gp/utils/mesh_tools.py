#!/usr/bin/env python
# encoding: utf-8

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
