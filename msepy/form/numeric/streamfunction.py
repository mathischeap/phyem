# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import numpy as np
from tools.dds.region_wise_structured import DDSRegionWiseStructured


class MsePyRootFormNumericStreamFunction(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._freeze()

    def ___parse_t___(self, t):
        if t is None:
            t = self._f.cochain.newest
        else:
            pass
        return t

    def ___decide_type___(self):
        """"""
        indicator = self._f.space.abstract.indicator
        if indicator == 'Lambda':
            space = self._f.space.abstract
            m, n, k = space.m, space.n, space.k
            if m == n == 2 and k == 1:
                return 2, 'vector'            # 2d-vector
            else:
                raise Exception(
                    f"cannot compute streamfunction for {self._f.space} or not implemented.")
        else:
            raise Exception(
                f"cannot compute streamfunction for {self._f.space} or not implemented.")

    def __call__(self, **kwargs):
        """Use the default method (and data structure) to compute the streamfunction."""
        return self.rws(**kwargs)

    def rws(self, t=None, ddf=5):
        """Express the stream-function at time `t` as region-wise structured.

        Parameters
        ----------
        t
        ddf

        Returns
        -------

        """
        t = self.___parse_t___(t)
        ndim, dtype = self.___decide_type___()
        linspace = np.linspace(-1, 1, ddf)
        grid_xi_et_sg = [linspace for _ in range(ndim)]
        xyz, value = self._f.reconstruct(t, *grid_xi_et_sg)

        if (ndim, dtype) == (2, 'vector'):   # 2d-vector
            x, y = xyz
            u, v = value
            x, y, u, v = self._f.mesh._regionwsie_stack(x, y, u, v)
            if len(x) == 1:  # only one region!
                region_key = list(x.keys())[0]
                sf0 = self._compute_1rw_2d_streamfunction_from_xy_uv(
                    x[region_key], y[region_key], u[region_key], v[region_key]
                )
                sf = {region_key: sf0}
                return DDSRegionWiseStructured([x, y], [sf])
            else:
                raise NotImplementedError((ndim, dtype))
        else:
            raise NotImplementedError()

    def _compute_1rw_2d_streamfunction_from_xy_uv(
            self,
            x, y, u, v,
            reference_corner=(0, 0),     # start with this corner,
                                         # for example, (0, 0) means the r-, s- corner of the region
            reference_streamfunction=0,  # start with this value
    ):
        """"""
        assert x.ndim == y.ndim == u.ndim == v.ndim == 2, f"data dimensions wrong."
        assert x.shape == y.shape == u.shape == v.shape, f"data shape wrong."
        sp0, sp1 = x.shape
        sf = np.ones_like(x)
        if reference_corner == (0, 0):  # start with r-, s- corner of the region.
            # col #0
            sf[0, 0] = reference_streamfunction
            for i in range(1, sp0):
                sf_start = sf[i-1, 0]

                sx, sy = x[i-1, 0], y[i-1, 0]
                ex, ey = x[i, 0], y[i, 0]

                dx = ex - sx
                dy = ey - sy

                su, sv = u[i-1, 0], v[i-1, 0]
                eu, ev = u[i, 0], v[i, 0]

                mu = (su + eu) / 2
                mv = (sv + ev) / 2

                d_sf_0 = mu * dy
                d_sf_1 = - mv * dx

                sf_end = sf_start + d_sf_0 + d_sf_1

                sf[i, 0] = sf_end

            # col [1:]
            for j in range(1, sp1):
                # node [j, 0]
                sf_start = sf[0, j-1]

                sx, sy = x[0, j-1], y[0, j-1]
                ex, ey = x[0, j], y[0, j]

                dx = ex - sx
                dy = ey - sy

                su, sv = u[0, j-1], v[0, j-1]
                eu, ev = u[0, j], v[0, j]

                mu = (su + eu) / 2
                mv = (sv + ev) / 2

                d_sf_0 = mu * dy
                d_sf_1 = - mv * dx

                sf_end = sf_start + d_sf_0 + d_sf_1

                sf[0, j] = sf_end

                # row [1:]
                for i in range(1, sp0):
                    sf_start = sf[i-1, j]

                    sx, sy = x[i-1, j], y[i-1, j]
                    ex, ey = x[i, j], y[i, j]

                    dx = ex - sx
                    dy = ey - sy

                    su, sv = u[i-1, j], v[i-1, j]
                    eu, ev = u[i, j], v[i, j]

                    mu = (su + eu) / 2
                    mv = (sv + ev) / 2

                    d_sf_0 = mu * dy
                    d_sf_1 = - mv * dx

                    sf_end = sf_start + d_sf_0 + d_sf_1

                    sf[i, j] = sf_end

            return sf

        else:
            raise NotImplementedError()
