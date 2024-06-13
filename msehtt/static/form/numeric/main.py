# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from src.config import RANK, MASTER_RANK, COMM

from tools.frozen import Frozen
from msehtt.static.form.numeric.tsp import MseHtt_Form_Numeric_TimeSpaceProperties
from tools.dds.region_wise_structured import DDSRegionWiseStructured


class MseHtt_Form_Numeric(Frozen):
    """"""

    def __init__(self, f):
        """"""
        self._f = f
        self._tsp = None
        self._cache_t_ = id(self)
        self._cache_itp_ = None
        self._export = None
        self._freeze()

    @property
    def tsp(self):
        if self._tsp is None:
            self._tsp = MseHtt_Form_Numeric_TimeSpaceProperties(self._f)
        return self._tsp

    def rws(self, t, ddf=1):
        """Return a dds-rws instance in the master rank."""
        density = int(9 * ddf)
        if density < 3:
            density = 3
        elif density > 35:
            density = 35
        else:
            pass
        linspace = np.linspace(-1, 1, density)
        xyz, value = self._f[t].reconstruct(linspace, linspace, ravel=False)
        XYZ = list()
        VAL = list()
        for _ in xyz:
            XYZ.append(self._merge_dict_(_, root=MASTER_RANK))
        for _ in value:
            VAL.append(self._merge_dict_(_, root=MASTER_RANK))

        if RANK == MASTER_RANK:
            return DDSRegionWiseStructured(XYZ, VAL)
        else:
            pass

    @staticmethod
    def _merge_dict_(data, root=MASTER_RANK):
        """"""
        assert isinstance(data, dict)
        DATA = COMM.gather(data, root=root)
        if RANK == root:
            data = {}
            for _ in DATA:
                data.update(_)
            return data
        else:
            return None

    @property
    def dtype(self):
        """"""
        space_indicator = self._f.space.str_indicator
        if space_indicator in ('m2n2k2', 'm2n2k0'):
            dtype = '2d-scalar'
        elif space_indicator in ('m2n2k1_inner', 'm2n2k1_outer'):
            dtype = '2d-vector'
        elif space_indicator in ('m3n3k0', 'm3n3k3'):
            dtype = '3d-scalar'
        elif space_indicator in ('m3n3k1', 'm3n3k2'):
            dtype = '3d-vector'
        else:
            raise NotImplementedError()
        return dtype

    # ----------- methods --------------------------------------------------------------
    def _interpolate_(self, t=None, ddf=1, data_only=False):
        """Use the solution of self._f at time `t` to make interpolations.

        Note that the output interpolation is rank-wise, so it only returns reasonable results when coordinates
        are in elements of that rank.

        """
        if t is None:
            t = self._f.cochain.newest
        else:
            t = self._f.cochain._parse_t(t)

        assert t is not None, f"I must have a t!"

        if data_only:
            pass
        else:
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
        elif ndim == 3:
            rc = form_at_t.reconstruct(linspace, linspace, linspace, ravel=True)
        else:
            raise NotImplementedError()

        dtype = self.dtype

        if dtype == '2d-scalar':
            xy, v = rc
            x, y = xy
            v = v[0]
            X, Y = list(), list()
            V = list()
            for e in x:
                X.extend(x[e])
                Y.extend(y[e])
                V.extend(v[e])
            if data_only:
                return dtype, (X, Y), (V,)
            else:
                interp = NearestNDInterpolator(
                    list(zip(X, Y)), V
                )
                self._cache_itp_ = ['2d-scalar', (interp, )]
                # do not remove (.,) since it shows we are getting something representing a scalar.

        elif dtype == '2d-vector':
            xy, uv = rc
            x, y = xy
            u, v = uv
            X, Y = list(), list()
            U, V = list(), list()
            for e in x:
                X.extend(x[e])
                Y.extend(y[e])
                U.extend(u[e])
                V.extend(v[e])
            if data_only:
                return dtype, (X, Y), (U, V)
            else:
                interp_u = NearestNDInterpolator(
                    list(zip(X, Y)), U
                )
                interp_v = NearestNDInterpolator(
                    list(zip(X, Y)), V
                )
                self._cache_itp_ = ['2d-vector', (interp_u, interp_v)]

        elif dtype == '3d-scalar':
            xyz, v = rc
            x, y, z = xyz
            v = v[0]
            X, Y, Z = list(), list(), list()
            V = list()
            for e in x:
                X.extend(x[e])
                Y.extend(y[e])
                Z.extend(z[e])
                V.extend(v[e])
            if data_only:
                return dtype, (X, Y, Z), (V,)
            else:
                interp = NearestNDInterpolator(
                    list(zip(X, Y, Z)), V
                )
                self._cache_itp_ = ['3d-scalar', (interp, )]
                # do not remove (.,) since it shows we are getting something representing a scalar.

        elif dtype == '3d-vector':
            xyz, uvw = rc
            x, y, z = xyz
            u, v, w = uvw
            X, Y, Z = list(), list(), list()
            U, V, W = list(), list(), list()
            for e in x:
                X.extend(x[e])
                Y.extend(y[e])
                Z.extend(z[e])
                U.extend(u[e])
                V.extend(v[e])
                W.extend(w[e])
            if data_only:
                return dtype, (X, Y, Z), (U, V, W)
            else:
                interp_u = NearestNDInterpolator(list(zip(X, Y, Z)), U)
                interp_v = NearestNDInterpolator(list(zip(X, Y, Z)), V)
                interp_w = NearestNDInterpolator(list(zip(X, Y, Z)), W)
                self._cache_itp_ = ['3d-vector', (interp_u, interp_v, interp_w)]

        else:
            raise NotImplementedError()

        return self._cache_itp_
