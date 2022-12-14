#!/usr/bin/env python
# encoding: utf-8

# Plot function on the mesh
def plot_function(x, y, z, triangles, function):
    from mayavi import mlab

    mlab.figure()
    mlab.triangular_mesh(x, y, z, triangles, scalars=function)
    v_options = {'mode': 'sphere',
                 'scale_factor': 1e-2, }
    mlab.points3d(x[0], y[0], z[0], **v_options)
