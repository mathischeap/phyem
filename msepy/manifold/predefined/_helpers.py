# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
from tools.frozen import Frozen
from numpy import ones_like


class _LinearTransformation(Frozen):
    """
    [0, 1]^n -> [x0, x1] x [y0, y1] x ...

    x0, x1, y0, y1 ... = * xb_yb_zb

    len(xb_yb_zb) = 2 * n

    """

    def __init__(self, *xb_yb_zb):
        """"""
        assert len(xb_yb_zb) % 2 == 0 and len(xb_yb_zb) >= 2, f"axis bounds must be even number and > 2."
        self._low_bounds = list()
        self._delta = list()
        for i in range(0, len(xb_yb_zb), 2):
            lb, ub = xb_yb_zb[i], xb_yb_zb[i+1]  # with `i` being axis
            assert ub > lb, f"lb={lb}, ub={ub} of {i}th axis is wrong. Must have lb < up."
            self._low_bounds.append(lb)
            self._delta.append(ub - lb)

        self._freeze()

    def mapping(self, *rst):
        """"""
        assert len(rst) == len(self._low_bounds), f"rst dimensions wrong."
        x = list()
        for i, r in enumerate(rst):
            lb = self._low_bounds[i]
            delta = self._delta[i]
            x.append(r * delta + lb)
        return x

    def Jacobian_matrix(self, *rst):
        I = len(self._low_bounds)
        assert len(rst) == I, f"rst dimensions wrong."
        J = [[0 for _ in range(I)] for _ in range(I)]
        r = rst[0]
        for i in range(I):
            J[i][i] = self._delta[i] * ones_like(r)   # important, must do ones_like.
        return tuple(J)
