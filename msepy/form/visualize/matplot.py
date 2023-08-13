# -*- coding: utf-8 -*-
r"""
"""
import matplotlib.pyplot as plt
import numpy as np
from tools.frozen import Frozen
from tools.matplot.plot import plot
from tools.matplot.contour import contour, contourf
from tools.matplot.quiver import quiver


class MsePyRootFormVisualizeMatplot(Frozen):
    """"""
    def __init__(self, rf):
        """"""
        self._f = rf
        self._mesh = rf.mesh
        self._freeze()

    def __call__(self, *args, title=None, **kwargs):
        """Call the default plotter coded in this module as well."""
        abs_sp = self._f.space.abstract
        m = abs_sp.m
        n = abs_sp.n
        k = abs_sp.k
        orientation = abs_sp.orientation

        if title is None:
            title = r'$t=%.3f$' % self._f.visualize._t  # this visualize is for cochain @ t
        else:
            pass

        if m == n == 2 and k == 1:
            return getattr(self, f'_m{m}_n{n}_k{k}_{orientation}')(*args, title=title, **kwargs)
        else:
            return getattr(self, f'_m{m}_n{n}_k{k}')(*args, title=title, **kwargs)

    def _m1_n1_k0(
            self, sampling_factor=1,
            figsize=(10, 6),
            color='k',
            **kwargs
    ):
        """"""
        samples = 500 * sampling_factor
        samples = int((np.ceil(samples / self._mesh.elements._num))**(1/self._mesh.m))
        if samples > 100:
            samples = 100
        elif samples < 5:
            samples = 5
        else:
            samples = int(samples)

        linspace = np.linspace(-1, 1, samples)
        t = self._f.visualize._t
        x, v = self._f[t].reconstruct(linspace)

        x = x[0]
        v = v[0]
        num_lines = len(x)  # also num elements
        fig = plot(
            x, v, num_lines=num_lines, xlabel='$x$', labels=False, styles='-',
            figsize=figsize, colors=color,
            **kwargs
        )
        return fig

    def _m1_n1_k1(self, *args, **kwargs):
        """"""
        return self._m1_n1_k0(*args, **kwargs)

    def _m2_n2_k0(
            self, sampling_factor=1,
            plot_type='contourf',
            **kwargs
    ):
        """"""
        samples = 10000 * sampling_factor
        samples = int((np.ceil(samples / self._mesh.elements._num))**(1/self._mesh.m))
        if samples > 75:
            samples = 75
        elif samples < 5:
            samples = 5
        else:
            samples = int(samples)

        xi_et = np.linspace(-1, 1, samples)
        t = self._f.visualize._t
        xy, v = self._f[t].reconstruct(xi_et, xi_et)  # ravel=False by default
        x, y = xy
        x, y, v = self._mesh._regionwsie_stack(x, y, v[0])

        if plot_type == 'contourf':
            fig = contourf(x, y, v, **kwargs)
        elif plot_type == 'contour':
            fig = contour(x, y, v, **kwargs)
        else:
            raise Exception()

        return fig

    def _m2_n2_k1_inner(
            self, sampling_factor=1,
            plot_type='contourf',
            saveto=None,
            title=None,
            title_components=None,
            **kwargs
    ):
        """"""
        samples = 10000 * sampling_factor
        samples = int((np.ceil(samples / self._mesh.elements._num))**(1/self._mesh.m))

        if samples > 75:
            samples = 75
        elif samples < 5:
            samples = 5
        else:
            samples = int(samples)

        xi_et = np.linspace(-1, 1, samples)
        t = self._f.visualize._t
        xy, uv = self._f[t].reconstruct(xi_et, xi_et)  # ravel=False by default
        x, y = xy
        u, v = uv
        x, y, u, v = self._mesh._regionwsie_stack(x, y, u, v)
        if plot_type in ('contourf', 'contour'):
            if saveto is None:
                saveto_x = None
                saveto_y = None
            else:
                saveto0, saveto1 = saveto.split('.')
                saveto_x = saveto0 + '_x' + '.' + saveto1
                saveto_y = saveto0 + '_y' + '.' + saveto1

            if title_components is None:
                title_x = '$x$-component'
                title_y = '$y$-component'
            else:
                title_x, title_y = title_components

            if plot_type == 'contourf':
                fig = [
                    contourf(x, y, u, title=title_x, saveto=saveto_x, **kwargs),
                    contourf(x, y, v, title=title_y, saveto=saveto_y, **kwargs)
                ]
            elif plot_type == 'contour':
                fig = [
                    contour(x, y, u, title=title_x, saveto=saveto_x, **kwargs),
                    contour(x, y, v, title=title_y, saveto=saveto_y, **kwargs)
                ]
            else:
                raise Exception()

        elif plot_type == "quiver":
            fig = self._quiver(x, y, u, v, saveto=saveto, **kwargs)
        else:
            raise Exception()

        if title is None:
            pass
        else:
            plt.suptitle(title)

        return fig

    def _m2_n2_k1_outer(self, **kwargs):
        """"""
        return self._m2_n2_k1_inner(**kwargs)

    def _m2_n2_k2(self, **kwargs):
        """"""
        return self._m2_n2_k0(**kwargs)

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
