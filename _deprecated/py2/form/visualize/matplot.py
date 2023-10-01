# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from tools.matplot.contour import contour, contourf
from tools.matplot.quiver import quiver

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
plt.rcParams.update({
    "font.family": "DejaVu sans",
})


class MseHyPy2RootFormVisualizeMatplot(Frozen):
    """"""
    def __init__(self, rf):
        """"""
        self._f = rf
        self._mesh = rf.mesh
        self._freeze()

    def __call__(self, title=None, **kwargs):
        """Call the default plotter coded in this module as well."""
        abs_sp = self._f.space.abstract
        k = abs_sp.k

        if title is None:
            title = r'$t=%.3f$' % self._f.visualize._t  # this visualize is for cochain @ t
        else:
            pass

        indicator = self._f._space.abstract.indicator

        if indicator in ('Lambda', ):
            return getattr(self, f'_Lambda_k{k}')(title=title, **kwargs)

        else:
            raise NotImplementedError(f"msepy matplot not implemented for {indicator}")

    def _Lambda_k0(
            self, sampling_factor=1,
            plot_type='contourf', show_mesh=False, saveto=None,
            **kwargs
    ):
        """"""
        t, g = self._f.visualize._t, self._f.visualize._g
        representative = self._f.mesh[g]
        samples = 10000 * sampling_factor
        samples = int((np.ceil(samples / len(representative)))**(1/2))
        if samples > 75:
            samples = 75
        elif samples < 5:
            samples = 5
        else:
            samples = int(samples)

        xi_et = np.linspace(-1, 1, samples)
        xy, v = self._f[(t, g)].reconstruct(xi_et, xi_et)  # ravel=False by default
        x, y = xy
        v = v[0]

        if show_mesh:
            intermediate = True
        else:
            intermediate = False

        if plot_type == 'contourf':
            fig = contourf(x, y, v, intermediate=intermediate, saveto=saveto, **kwargs)
        elif plot_type == 'contour':
            fig = contour(x, y, v, intermediate=intermediate, saveto=saveto, **kwargs)
        else:
            raise Exception()

        if show_mesh:
            mesh_data = self._mesh[self._f.visualize._g].visualize._make_mesh_data(
                sampling_factor=sampling_factor)
            for i in mesh_data:
                for line in mesh_data[i]:
                    plt.plot(*line, linewidth=0.5, color='gray')
            if saveto is None:
                from src.config import _setting
                matplotlib.use('TkAgg')
                plt.show(block=_setting['block'])
            else:
                plt.savefig(saveto, bbox_inches='tight', pad_inches=0, dpi=200)
                plt.close()

            plt.close()
        else:
            pass

        return fig

    def _Lambda_k1(
            self, sampling_factor=1,
            plot_type='contourf',
            saveto=None,
            title=None,
            title_components=None,
            **kwargs
    ):
        """"""
        t, g = self._f.visualize._t, self._f.visualize._g
        representative = self._f.mesh[g]
        samples = 10000 * sampling_factor
        samples = int((np.ceil(samples / len(representative)))**(1/2))

        if samples > 75:
            samples = 75
        elif samples < 5:
            samples = 5
        else:
            samples = int(samples)

        xi_et = np.linspace(-0.95, 1, samples)
        xy, uv = self._f[(t, g)].reconstruct(xi_et, xi_et)  # ravel=False by default
        x, y = xy
        u, v = uv
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

    def _Lambda_k2(
        self, sampling_factor=1,
        plot_type='contourf', show_mesh=False, saveto=None,
        **kwargs
    ):
        t, g = self._f.visualize._t, self._f.visualize._g
        representative = self._f.mesh[g]
        samples = 10000 * sampling_factor
        samples = int((np.ceil(samples / len(representative)))**(1/2))
        if samples > 75:
            samples = 75
        elif samples < 5:
            samples = 5
        else:
            samples = int(samples)

        xi_et = np.linspace(-0.95, 1, samples)
        xy, v = self._f[(t, g)].reconstruct(xi_et, xi_et)  # ravel=False by default
        x, y = xy
        v = v[0]

        if show_mesh:
            intermediate = True
        else:
            intermediate = False

        if plot_type == 'contourf':
            fig = contourf(x, y, v, intermediate=intermediate, saveto=saveto, **kwargs)
        elif plot_type == 'contour':
            fig = contour(x, y, v, intermediate=intermediate, saveto=saveto, **kwargs)
        else:
            raise Exception()

        if show_mesh:
            mesh_data = self._mesh[self._f.visualize._g].visualize._make_mesh_data(
                sampling_factor=sampling_factor)
            for i in mesh_data:
                for line in mesh_data[i]:
                    plt.plot(*line, linewidth=0.75, color='lightgray')
            if saveto is None:
                from src.config import _setting
                matplotlib.use('TkAgg')
                plt.show(block=_setting['block'])
            else:
                plt.savefig(saveto, bbox_inches='tight', pad_inches=0, dpi=200)
                plt.close()

            plt.close()
        else:
            pass

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
