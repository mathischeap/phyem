# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
@time: 11/26/2022 2:56 PM
"""

from numpy import sin, pi, cos, ones_like

import warnings
from src.tools.frozen import Frozen


class CrazyMeshCurvatureWarning(UserWarning):
    pass


def crazy(mf, bounds=None, c=0, periodic=False):
    """"""
    assert mf.esd == mf.ndim, f"crazy mesh only works for manifold.ndim == embedding space dimensions."
    esd = mf.esd
    if bounds is None:
        bounds = [(0, 1) for _ in range(esd)]
    else:
        assert len(bounds) == esd, f"bounds={bounds} dimensions wrong."

    rm0 = _MesPyRegionCrazyMapping(bounds, c, esd)

    if periodic:
        region_map = {
            0: [0 for _ in range(2 * esd)],    # region #0
        }
    else:
        region_map = {
            0: [None for _ in range(2 * esd)],    # region #0
        }

    mapping_dict = {
        0: rm0.mapping,  # region #0
    }

    Jacobian_matrix_dict = {
        0: rm0.Jacobian_matrix
    }

    if c == 0:
        mtype = {'indicator': 'Linear', 'parameters': []}
        for i, lb_ub in enumerate(bounds):
            xyz = 'xyz'[i]
            lb, ub = lb_ub
            d = str(round(ub - lb, 5))  # do this to round off the truncation error.
            mtype['parameters'].append(xyz + d)
    else:
        mtype = None  # this is a unique region. Its metric does not like any other.

    mtype_dict = {
        0: mtype
    }

    return region_map, mapping_dict, Jacobian_matrix_dict, mtype_dict


class _MesPyRegionCrazyMapping(Frozen):

    def __init__(self, bounds, c, esd):
        for i, bs in enumerate(bounds):
            assert len(bs) == 2 and all([isinstance(_, (int, float)) for _ in bs]), f"bounds[{i}]={bs} is illegal."
            lb, up = bs
            assert lb < up, f"bounds[{i}]={bs} is illegal."
        assert isinstance(c, (int, float)), f"={c} is illegal, need to be a int or float. Ideally in [0, 0.3]."

        if not (0 <= c <= 0.3):
            warnings.warn(f"c={c} is not good. Ideally, c in [0, 0.3].", CrazyMeshCurvatureWarning)

        self._bounds = bounds
        self._c = c
        self._esd = esd
        self._freeze()

    def mapping(self, *rst):
        """ `*rst` be in [0, 1]. """
        assert len(rst) == self._esd, f"amount of inputs wrong."

        if self._esd == 1:
            r = rst[0]
            a, b = self._bounds[0]
            x = (b - a) * (r + 0.5 * self._c * sin(2 * pi * r)) + a
            return [x]

        elif self._esd == 2:

            r, s = rst
            a, b = self._bounds[0]
            c, d = self._bounds[1]
            if self._c == 0:
                x = (b - a) * r + a
                y = (d - c) * s + c
            else:
                x = (b - a) * (r + 0.5 * self._c * sin(2 * pi * r) * sin(2 * pi * s)) + a
                y = (d - c) * (s + 0.5 * self._c * sin(2 * pi * r) * sin(2 * pi * s)) + c
            return x, y

        elif self._esd == 3:
            r, s, t = rst
            a, b = self._bounds[0]
            c, d = self._bounds[1]
            e, f = self._bounds[2]

            if self._c == 0:
                x = (b - a) * r + a
                y = (d - c) * s + c
                z = (f - e) * t + e

            else:
                x = (b - a) * (r + 0.5 * self._c *
                               sin(2 * pi * r) *
                               sin(2 * pi * s) *
                               sin(2 * pi * t)) + a
                y = (d - c) * (s + 0.5 * self._c *
                               sin(2 * pi * r) *
                               sin(2 * pi * s) *
                               sin(2 * pi * t)) + c
                z = (f - e) * (t + 0.5 * self._c *
                               sin(2 * pi * r) *
                               sin(2 * pi * s) *
                               sin(2 * pi * t)) + e

            return x, y, z

        else:
            raise NotImplementedError()

    def Jacobian_matrix(self, *rst):
        """ r, s, t be in [0, 1]. """
        assert len(rst) == self._esd, f"amount of inputs wrong."

        if self._esd == 1:
            r = rst[0]
            a, b = self._bounds[0]
            if self._c == 0:
                xr = (b - a) * ones_like(r)
            else:
                xr = (b - a) + (b - a) * 2 * pi * 0.5 * self._c * cos(2 * pi * r)
            return [[xr]]

        elif self._esd == 2:
            r, s = rst
            
            a, b = self._bounds[0]
            c, d = self._bounds[1]
            if self._c == 0:
                xr = (b - a) * ones_like(r)
                xs = 0
                yr = 0
                ys = (d - c) * ones_like(r)

            else:
                xr = (b - a) + (b - a) * 2 * pi * 0.5 * self._c * cos(2 * pi * r) * sin(2 * pi * s)
                xs = (b - a) * 2 * pi * 0.5 * self._c * sin(2 * pi * r) * cos(2 * pi * s)
                yr = (d - c) * 2 * pi * 0.5 * self._c * cos(2 * pi * r) * sin(2 * pi * s)
                ys = (d - c) + (d - c) * 2 * pi * 0.5 * self._c * sin(2 * pi * r) * cos(2 * pi * s)
            return ((xr, xs),
                    (yr, ys))

        elif self._esd == 3:

            r, s, t = rst
            a, b = self._bounds[0]
            c, d = self._bounds[1]
            e, f = self._bounds[2]

            if self._c == 0:
                xr = (b - a) * ones_like(r)
                xs = 0  # np.zeros_like(r)
                xt = 0
    
                yr = 0
                ys = (d - c) * ones_like(r)
                yt = 0
    
                zr = 0
                zs = 0
                zt = (f - e) * ones_like(r)
            else:
                xr = (b - a) + (b - a) * 2 * pi * 0.5 * self._c * cos(2 * pi * r) * sin(2 * pi * s) * sin(
                    2 * pi * t)
                xs = (b - a) * 2 * pi * 0.5 * self._c * sin(2 * pi * r) * cos(2 * pi * s) * sin(
                    2 * pi * t)
                xt = (b - a) * 2 * pi * 0.5 * self._c * sin(2 * pi * r) * sin(2 * pi * s) * cos(
                    2 * pi * t)
    
                yr = (d - c) * 2 * pi * 0.5 * self._c * cos(2 * pi * r) * sin(2 * pi * s) * sin(
                    2 * pi * t)
                ys = (d - c) + (d - c) * 2 * pi * 0.5 * self._c * sin(2 * pi * r) * cos(2 * pi * s) * sin(
                    2 * pi * t)
                yt = (d - c) * 2 * pi * 0.5 * self._c * sin(2 * pi * r) * sin(2 * pi * s) * cos(
                    2 * pi * t)
    
                zr = (f - e) * 2 * pi * 0.5 * self._c * cos(2 * pi * r) * sin(2 * pi * s) * sin(
                    2 * pi * t)
                zs = (f - e) * 2 * pi * 0.5 * self._c * sin(2 * pi * r) * cos(2 * pi * s) * sin(
                    2 * pi * t)
                zt = (f - e) + (f - e) * 2 * pi * 0.5 * self._c * sin(2 * pi * r) * sin(2 * pi * s) * cos(
                    2 * pi * t)
    
            return [(xr, xs, xt),
                    (yr, ys, yt),
                    (zr, zs, zt)]
