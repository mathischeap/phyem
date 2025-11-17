# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from phyem.src.config import RANK, MASTER_RANK, COMM, SIZE
from phyem.tools.matplot.scatter import scatter


def quick_visualizer_m2n2k0(f, t, ddf=1, **kwargs):
    r""""""
    total_num_elements = f.tpm.composition.num_global_elements
    density = 5000 * ddf
    if density < 500:
        density = 500
    elif density > 10000:
        density = 10000
    else:
        density = int(density)

    linspace = int((density / total_num_elements)**0.5) + 2
    xi_et = np.linspace(-1, 1, linspace)
    xi_et = (xi_et[1:] + xi_et[:-1]) / 2

    Rc = f[t].reconstruct(xi_et, xi_et, ravel=True)
    xy, u = Rc
    x, y = xy
    u = u[0]

    x = COMM.gather(x, root=MASTER_RANK)
    y = COMM.gather(y, root=MASTER_RANK)
    u = COMM.gather(u, root=MASTER_RANK)

    if RANK != MASTER_RANK:
        return
    else:
        pass

    X, Y, U = {}, {}, {}
    for i in range(SIZE):
        X.update(x[i])
        Y.update(y[i])
        U.update(u[i])

    return scatter(X, Y, U, xlabel='$x$', ylabel='$y$', **kwargs)


def quick_visualizer_m2n2k1(f, t, ddf=1, title=None, **kwargs):
    r""""""
    total_num_elements = f.tpm.composition.num_global_elements
    density = 5000 * ddf
    if density < 500:
        density = 500
    elif density > 10000:
        density = 10000
    else:
        density = int(density)

    linspace = int((density / total_num_elements)**0.5) + 2
    xi_et = np.linspace(-1, 1, linspace)
    xi_et = (xi_et[1:] + xi_et[:-1]) / 2

    Rc = f[t].reconstruct(xi_et, xi_et, ravel=True)
    xy, uv = Rc
    x, y = xy
    u, v = uv

    x = COMM.gather(x, root=MASTER_RANK)
    y = COMM.gather(y, root=MASTER_RANK)
    u = COMM.gather(u, root=MASTER_RANK)
    v = COMM.gather(v, root=MASTER_RANK)

    if RANK != MASTER_RANK:
        return
    else:
        pass

    X, Y, U, V = {}, {}, {}, {}
    for i in range(SIZE):
        X.update(x[i])
        Y.update(y[i])
        U.update(u[i])
        V.update(v[i])

    return (
        scatter(X, Y, U, title=title + ' $x$-component', xlabel='$x$', ylabel='$y$', **kwargs),
        scatter(X, Y, V, title=title + ' $y$-component', xlabel='$x$', ylabel='$y$', **kwargs),
    )


def quick_visualizer_m2n2k2(f, t, ddf=1, **kwargs):
    r""""""
    return quick_visualizer_m2n2k0(f, t, ddf=ddf, **kwargs)
