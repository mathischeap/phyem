# -*- coding: utf-8 -*-
"""
phyem@RAM-EEMCS-UT
Yi Zhang
"""

import numpy as np
import sys
if './' not in sys.path:
    sys.path.append('./')

from tools.frozen import Frozen
from tools.matplot.plot import plot
from tools.matplot.contour import contour, contourf


class MsePyRootFormVisualizeMatplot(Frozen):
    """"""
    def __init__(self, rf):
        """"""
        self._f = rf
        self._mesh = rf.mesh
        self._freeze()

    def __call__(self, *args, **kwargs):
        """Call the default plotter coded in this module as well."""
        abs_sp = self._f.space.abstract
        m = abs_sp.m
        n = abs_sp.n
        k = abs_sp.k
        orientation = abs_sp.orientation

        if m == n == 2 and k == 1:
            return getattr(self, f'_m{m}_n{n}_k{k}_{orientation}')(*args, **kwargs)
        else:
            return getattr(self, f'_m{m}_n{n}_k{k}')(*args, **kwargs)

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

        x = x[0].T
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

        if plot_type == 'contourf':
            fig = [contourf(x, y, u, **kwargs), contourf(x, y, v, **kwargs)]
        elif plot_type == 'contour':
            fig = [contour(x, y, u, **kwargs), contour(x, y, v, **kwargs)]
        else:
            raise Exception()

        return fig

    def _m2_n2_k1_outer(self, **kwargs):
        """"""
        return self._m2_n2_k1_inner(**kwargs)

    def _m2_n2_k2(self, **kwargs):
        """"""
        return self._m2_n2_k0(**kwargs)
