# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.frozen import Frozen
from phyem.tools.matplot.contour import contour, contourf
from phyem.tools.matplot.quiver import quiver


class MsePyMeshVisualizeTarget(Frozen):
    """"""

    def __init__(self, mesh):
        """"""
        self._mesh = mesh
        self._freeze()

    def __call__(self, function, sampling_factor=1, **kwargs):
        """"""
        samples = int(10 * sampling_factor)
        if samples < 5:
            samples = 5
        elif samples > 100:
            samples = 100
        else:
            pass
        n = self._mesh.n
        coo = [np.linspace(-1, 1, samples) for _ in range(n)]
        coo = np.meshgrid(*coo, indexing='ij')
        coo = self._mesh.ct.mapping(*coo)
        val = function(*coo)
        if n == 2 and len(val) == 1:
            # we are plotting a scalar on a 2d mesh
            self._plot_2d_scalar_on_mesh(coo, val, **kwargs)
        elif n == 2 and len(val) == 2:
            # we are plotting a vector on a 2d mesh
            self._plot_2d_vector_on_mesh(coo, val, **kwargs)
        else:
            raise NotImplementedError(f"cannot plot this function on {n}d-mesh.")

    def _plot_2d_scalar_on_mesh(
            self, coo, val,
            plot_type='contourf',
            **kwargs
    ):
        x, y = coo
        v = val[0]

        x, y, v = self._mesh._regionwsie_stack(x, y, v, axis=-1)

        if plot_type == 'contourf':
            fig = contourf(x, y, v, **kwargs)
        elif plot_type == 'contour':
            fig = contour(x, y, v, **kwargs)
        else:
            raise Exception()

        return fig

    def _plot_2d_vector_on_mesh(
            self, coo, val,
            plot_type='contourf',
            **kwargs
    ):
        """"""
        x, y = coo
        u, v = val

        x, y, u, v = self._mesh._regionwsie_stack(x, y, u, v, axis=-1)

        if plot_type == 'contourf':
            fig = [
                contourf(x, y, u, title='$x$-component', **kwargs),
                contourf(x, y, v, title='$y$-component', **kwargs)
            ]
        elif plot_type == 'contour':
            fig = [
                contour(x, y, u, title='$x$-component', **kwargs),
                contour(x, y, v, title='$y$-component', **kwargs)
            ]
        elif plot_type == "quiver":
            fig = self._quiver(x, y, u, v, **kwargs)
        else:
            raise Exception()

        return fig

    @staticmethod
    def _quiver(
            x, y, u, v, **kwargs
    ):
        """"""
        X = list()
        Y = list()
        U = list()
        V = list()
        for i in x:
            X.append(x[i])
            Y.append(y[i])
            U.append(u[i])
            V.append(v[i])

        U = np.array(U).ravel()
        V = np.array(V).ravel()
        X = np.array(X).ravel()
        Y = np.array(Y).ravel()
        return quiver(X, Y, U, V, **kwargs)
