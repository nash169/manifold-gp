import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def colormap_diverging(colormap, vmin, vmax, center_color=np.array([1., 1., 1., 0.]), res=1000, step=0.1):
    # values interval
    l = vmax - vmin

    # number of points in the first half
    n1 = int(np.ceil(res*abs(vmin)/l))

    # number of points in the second half
    n2 = int(np.floor(res*vmax/l))

    # starting colormap
    cmap = plt.get_cmap(colormap)

    # number of sampled colors in the first half
    p1 = np.arange(0, 0.5, step)

    # number of sampled color in the second half
    p2 = np.arange(0.5+step, 1.0+step, step)

    # number of points and rest within each step first half
    w1 = int(n1/len(p1))
    r1 = n1 - w1*len(p1)

    # number of points and rest within each step second half
    w2 = int(n2/len(p2))
    r2 = n2 - w2*len(p2)

    # rgba colors
    rgb = np.empty([0, 4])

    # reconstruct first half of the colormap
    for i in range(len(p1)-1):
        rgb = np.append(rgb, np.linspace(np.array(cmap(p1[i])), np.array(cmap(p1[i+1])), w1), axis=0)

    # converge to white till the half of the colormap
    rgb = np.append(rgb, np.linspace(np.array(cmap(p1[-1])), center_color, w1 + r1), axis=0)

    # diverge from white at the half of the colormap
    rgb = np.append(rgb, np.linspace(center_color, np.array(cmap(p2[0])), w2 + r2), axis=0)

    # reconstruct second half of the colormap
    for i in range(len(p2)-1):
        rgb = np.append(rgb, np.linspace(np.array(cmap(p2[i])), np.array(cmap(p2[i+1])), w2), axis=0)

    # generate colormap
    cmap = mcolors.ListedColormap(rgb)

    return cmap


def colormap_left(colormap, left_color=np.array([1., 1., 1., 0.]), res=1000, step=0.1):
    # starting colormap
    cmap = plt.get_cmap(colormap)

    # number of sampled colors in the first half
    p = np.arange(step, 1.0+step, step)

    # starting colormap
    cmap = plt.get_cmap(colormap)

    # number of points and rest
    w = int(res/len(p))
    r = res - w*len(p)

    # rgba colors
    rgb = np.empty([0, 4])

    # diverge from white
    rgb = np.append(rgb, np.linspace(left_color, np.array(cmap(p[0])), w + r), axis=0)

    # reconstruct colormap
    for i in range(len(p)-1):
        rgb = np.append(rgb, np.linspace(np.array(cmap(p[i])), np.array(cmap(p[i+1])), w), axis=0)

    # generate colormap
    cmap = mcolors.ListedColormap(rgb)

    return cmap


def colormap_right(colormap, right_color=np.array([1., 1., 1., 0.]), res=1000, step=0.1):
    # starting colormap
    cmap = plt.get_cmap(colormap)

    # number of sampled colors in the first half
    p = np.arange(0.0, 1.0, step)

    # starting colormap
    cmap = plt.get_cmap(colormap)

    # number of points and rest
    w = int(res/len(p))
    r = res - w*len(p)

    # rgba colors
    rgb = np.empty([0, 4])

    # reconstruct colormap
    for i in range(len(p)-1):
        rgb = np.append(rgb, np.linspace(np.array(cmap(p[i])), np.array(cmap(p[i+1])), w), axis=0)

    # converge to white
    rgb = np.append(rgb, np.linspace(np.array(cmap(p[-1])), right_color, w + r), axis=0)

    # generate colormap
    cmap = mcolors.ListedColormap(rgb)

    return cmap


def colorbar(im, fig, ax, pos="left", size="5%", pad=0.2, ticks=None):
    # colorbar location
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(pos, size=size, pad=pad)

    # create colorbar and set ticks
    cbar = fig.colorbar(im, cax=cax, ticks=ticks)

    # set location ticks
    cax.yaxis.set_ticks_position(pos)

    # remove colorbar frame
    cbar.outline.set_visible(False)

    # remove ticks
    cbar.set_ticks([])
    # cbar.ax.tick_params(size=0)


def beautify(fig, ax):
    # disable x ticks
    ax.axes.get_xaxis().set_visible(False)

    # disable y ticks
    ax.axes.get_yaxis().set_visible(False)

    # transparency
    fig.patch.set_visible(False)

    # invisible frame
    ax.axis('off')

    # equal axis
    ax.axis('equal')

    # remove margins
    fig.tight_layout()

    # for c in im.collections:
    #     c.set_edgecolor("face")


def plot_1D_mesh(fig, ax, vertices, edges, gradient):
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
