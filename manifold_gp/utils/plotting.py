import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def colormap_diverging(colormap, vmin, vmax, center_color=np.array([1., 1., 1., 0.]), res=1000, step=0.1):
    l = vmax - vmin

    n1 = int(np.ceil(res*abs(vmin)/l))
    n2 = int(np.floor(res*vmax/l))

    cmap = plt.get_cmap(colormap)
    p1 = np.arange(0, 0.5, step)
    p2 = np.arange(0.5+step, 1.0+step, step)
    w1 = int(n1/len(p1))
    r1 = n1 - w1*len(p1)
    w2 = int(n2/len(p2))
    r2 = n2 - w2*len(p2)

    rgb = np.empty([0, 4])

    for i in range(len(p1)-1):
        np.append(rgb, np.linspace(np.array(cmap(p1[i])), np.array(cmap(p1[i+1])), w1), axis=0)

    rgb = np.append(rgb, np.linspace(np.array(cmap(p1[-1])), center_color, w1 + r1), axis=0)
    rgb = np.append(rgb, np.linspace(center_color, np.array(cmap(p1[0])), w2 + r2), axis=0)

    for i in range(len(p2)-1):
        rgb = np.append(rgb, np.linspace(np.array(cmap(p2[i])), np.array(cmap(p2[i+1])), w2), axis=0)

    cmap = mcolors.ListedColormap(rgb)


def colorbar(fig, ax):
    pass


def beautify(fig, ax):
    # disable x ticks
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
