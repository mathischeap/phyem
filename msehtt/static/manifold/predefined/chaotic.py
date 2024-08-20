# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from numpy import sin, pi, cos, ones_like
from random import randint
from src.config import RANK, MASTER_RANK


___A___ = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
    [1, 0, 1, 0]
])

___invA___ = np.linalg.inv(___A___)


def chaotic(bounds=([0, 1], [0, 1]), c=0, periodic=False):
    r"""Mainly for test purpose."""
    assert RANK == MASTER_RANK, f"only initialize chaotic mesh in the master rank"

    if len(bounds) == 2:
        return crazy2d(bounds=bounds, c=c, periodic=periodic)
    elif len(bounds) == 3:
        return crazy3d(bounds=bounds, c=c, periodic=periodic)
    else:
        raise Exception


# ============ 2d =====================================================================


def crazy2d(bounds=([0, 1], [0, 1]), c=0, periodic=False):
    r""""""
    if periodic:
        raise NotImplementedError()
    else:
        pass

    # ----- there will be 9 regions --------------------------------------------
    low_x, upp_x = bounds[0]
    low_y, upp_y = bounds[1]

    total_mapping = TotalMapping2D(low_x, upp_x, low_y, upp_y, c)

    assert low_x < upp_x and low_y < upp_y, f"bounds = {bounds} wrong."

    X = np.linspace(low_x, upp_x, 4)
    Y = np.linspace(low_y, upp_y, 4)

    regions = {}
    for m in range(9):
        regions[m] = {
            'x': [],
            'y': [],
        }

    shift = [randint(0, 3) for _ in range(9)]
    # shift = [0 for _ in range(9)]

    for j in range(3):
        for i in range(3):
            m = i + j * 3
            xx = [X[i], X[i+1], X[i+1], X[i]]
            yy = [Y[j], Y[j], Y[j+1], Y[j+1]]

            sft = shift[m]

            if sft == 0:
                pass
            elif sft == 1:
                xx = [xx[1], xx[2], xx[3], xx[0]]
                yy = [yy[1], yy[2], yy[3], yy[0]]
            elif sft == 2:
                xx = [xx[2], xx[3], xx[0], xx[1]]
                yy = [yy[2], yy[3], yy[0], yy[1]]
            elif sft == 3:
                xx = [xx[3], xx[0], xx[1], xx[2]]
                yy = [yy[3], yy[0], yy[1], yy[2]]
            else:
                raise Exception()

            regions[m]['x'] = xx
            regions[m]['y'] = yy

    REGIONS = {}
    for m in range(9):
        A = ___invA___ @ np.array(regions[m]['x'])
        B = ___invA___ @ np.array(regions[m]['y'])
        REGIONS[m] = _Single_Map_(A, B, total_mapping)

    region_map = None        # the config method will parse the region map.
    periodic_setting = None

    return REGIONS, region_map, periodic_setting


class _Single_Map_(Frozen):
    r""""""
    def __init__(self, A, B, total_mapping):
        """
        It first maps [0, 1]^2 into a quad region (q, w). q, w can be computed by affine
        quad mapping. See `mapping`.

        Then the (q, w) region is mapping to a physical region with the total crazy mapping.

        Parameters
        ----------
        A
        B
        total_mapping
        """
        self._a1, self._a2, self._a3, self._a4 = A
        self._b1, self._b2, self._b3, self._b4 = B
        self._tm = total_mapping
        self._freeze()

    @property
    def ndim(self):
        """This is a 2d region."""
        return 2

    @property
    def etype(self):
        r"""The element made in this region can only be of this type."""
        if self._tm._c == 0:
            return 9
        else:
            return 'unique curvilinear quad'

    def mapping(self, r, s):
        """"""
        q = self._a1 + self._a2 * r + self._a3 * s + self._a4 * r * s
        w = self._b1 + self._b2 * r + self._b3 * s + self._b4 * r * s

        x, y = self._tm.mapping(q, w)

        return x, y

    def Jacobian_matrix(self, r, s):
        """"""

        qr = self._a2 + self._a4 * s
        qs = self._a3 + self._a4 * r

        wr = self._b2 + self._b4 * s
        ws = self._b3 + self._b4 * r

        q = self._a1 + self._a2 * r + self._a3 * s + self._a4 * r * s
        w = self._b1 + self._b2 * r + self._b3 * s + self._b4 * r * s

        JM = self._tm.Jacobian_matrix(q, w)

        xq, xw = JM[0]
        yq, yw = JM[1]

        xr = xq * qr + xw * wr
        xs = xq * qs + xw * ws

        yr = yq * qr + yw * wr
        ys = yq * qs + yw * ws

        return (
            [xr, xs],
            [yr, ys],
        )


class TotalMapping2D(Frozen):
    r""""""

    def __init__(self, a, b, c, d, deformation_factor):
        """"""
        self._abcd_ = (a, b, c, d)
        self._c = deformation_factor
        self._freeze()

    def mapping(self, r, s):
        r""""""
        a, b, c, d = self._abcd_
        if self._c == 0:
            x = (b - a) * r + a
            y = (d - c) * s + c
        else:
            x = (b - a) * (r + 0.5 * self._c * sin(2 * pi * r) * sin(2 * pi * s)) + a
            y = (d - c) * (s + 0.5 * self._c * sin(2 * pi * r) * sin(2 * pi * s)) + c
        return x, y

    def Jacobian_matrix(self, r, s):
        """ r, s, t be in [0, 1]. """
        a, b, c, d = self._abcd_

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


# ============ 3d =====================================================================


def crazy3d(bounds=([0, 1], [0, 1], [0, 1]), c=0, periodic=False):
    r""""""
    raise NotImplementedError()
