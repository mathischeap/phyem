# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from scipy.interpolate import NearestNDInterpolator

from tools.frozen import Frozen
from msehtt.static.form.numeric.tsp import MseHtt_Form_Numeric_TimeSpaceProperties


class MseHtt_Form_Numeric(Frozen):
    """"""

    def __init__(self, f):
        """"""
        self._f = f
        self._tsp = None
        self._cache_t_ = id(self)
        self._cache_itp_ = None
        self._freeze()

    @property
    def tsp(self):
        if self._tsp is None:
            self._tsp = MseHtt_Form_Numeric_TimeSpaceProperties(self._f)
        return self._tsp

    # ----------- methods --------------------------------------------------------------
    def _interpolate_(self, t=None, ddf=1):
        """Use the solution of self._f at time `t` to make interpolations.

        Note that the output interpolation is rank-wise, so it only returns reasonable results when coordinates
        are in elements of that rank.

        """
        if t is None:
            t = self._f.cochain.newest
        else:
            t = self._f.cochain._parse_t(t)

        assert t is not None, f"I must have a t!"

        if t == self._cache_t_:
            return self._cache_itp_
        else:
            self._cache_t_ = t

        density = int(17 * ddf)
        if density < 7:
            density = 7
        elif density > 31:
            density = 31
        else:
            pass
        linspace = np.linspace(-1, 1, density + 1)
        linspace = (linspace[1:] + linspace[:-1]) / 2

        form_at_t = self._f[t]
        ndim = self._f.space.n

        if ndim == 2:
            rc = form_at_t.reconstruct(linspace, linspace, ravel=True)
        else:
            raise NotImplementedError()

        space_indicator = self._f.space.str_indicator
        if space_indicator in ('m2n2k2', 'm2n2k0'):
            dtype = '2d-scalar'
        elif space_indicator in ('m2n2k1_inner', 'm2n2k1_outer'):
            dtype = '2d-vector'
        else:
            raise NotImplementedError()

        if dtype == '2d-scalar':
            xy, v = rc
            x, y = xy
            v = v[0]
            X = list()
            Y = list()
            V = list()
            for e in x:
                X.extend(x[e])
                Y.extend(y[e])
                V.extend(v[e])
            interp = NearestNDInterpolator(
                list(zip(X, Y)), V
            )
            self._cache_itp_ = ['2d-scalar', (interp, )]
            # do not remove (.,) since it shows we are getting something representing a scalar.

        elif dtype == '2d-vector':
            xy, uv = rc
            x, y = xy
            u, v = uv
            X = list()
            Y = list()
            U = list()
            V = list()
            for e in x:
                X.extend(x[e])
                Y.extend(y[e])
                U.extend(u[e])
                V.extend(v[e])
            interp_u = NearestNDInterpolator(
                list(zip(X, Y)), U
            )
            interp_v = NearestNDInterpolator(
                list(zip(X, Y)), V
            )
            self._cache_itp_ = ['2d-vector', (interp_u, interp_v)]

        else:
            raise NotImplementedError()

        return self._cache_itp_
