# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from numpy import ones_like
from tools.functions.space._2d.transfinite import TransfiniteMapping
from tools.functions.space._2d.geometrical_functions.parser import GeoFunc2Parser


class _LinearTransformation(Frozen):
    r"""
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
        self._mtype = None
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
        """"""
        I_ = len(self._low_bounds)
        assert len(rst) == I_, f"rst dimensions wrong."
        J = [[0 for _ in range(I_)] for _ in range(I_)]
        r = rst[0]
        for i in range(I_):
            J[i][i] = self._delta[i] * ones_like(r)  # important, must do ones_like.
        return tuple(J)

    @property
    def mtype(self):
        if self._mtype is None:
            parameters = list()
            for i, delta in enumerate(self._delta):
                axis = 'xyz'[i]
                parameters.append(axis + str(delta))

            self._mtype = {
                'indicator': 'Linear',
                'parameters': parameters
            }
        return self._mtype


class _Transfinite2(Frozen):
    r"""A wrapper of the 2d transfinite mapping.

     y          - (y1) +
     ^       _______________
     |      |               |
     |      |               |
     |    + |               | +
     | (x0) |               | (x1)
     |    - |               | -
     |      |               |
     |      |_______________|
     |          - (y0) +
     |_______________________> x

    The indices of `gamma` and `dgamma` are as above. And the directions of the
    mappings are indicated as well.

    """
    def __init__(self, geo_x0, geo_x1, geo_y0, geo_y1):
        """

        Parameters
        ----------
        geo_x0 :
            [str(geo_name), list(geo_parameters)]
        geo_x1
        geo_y0
        geo_y1

        """

        geo_x0 = GeoFunc2Parser(*geo_x0)
        geo_x1 = GeoFunc2Parser(*geo_x1)
        geo_y0 = GeoFunc2Parser(*geo_y0)
        geo_y1 = GeoFunc2Parser(*geo_y1)

        gamma = [
            geo_y0.gamma,
            geo_x1.gamma,
            geo_y1.gamma,
            geo_x0.gamma,
        ]
        d_gamma = [
            geo_y0.dgamma,
            geo_x1.dgamma,
            geo_y1.dgamma,
            geo_x0.dgamma,
        ]
        self._tf = TransfiniteMapping(gamma, d_gamma)
        self._geo_x0 = geo_x0
        self._geo_x1 = geo_x1
        self._geo_y0 = geo_y0
        self._geo_y1 = geo_y1
        self._mtype = None
        self._freeze()

    def mapping(self, r, s):
        """"""
        return self._tf.mapping(r, s)

    def Jacobian_matrix(self, r, s):
        """"""
        return (
            [self._tf.dx_dr(r, s), self._tf.dx_ds(r, s)],
            [self._tf.dy_dr(r, s), self._tf.dy_ds(r, s)]
        )

    def illustrate(self):
        """"""
        return self._tf.illustrate()

    @property
    def mtype(self):
        """"""
        if self._mtype is None:
            names = (
                self._geo_x0._name,
                self._geo_x1._name,
                self._geo_y0._name,
                self._geo_y1._name
            )
            if all([_ == 'straight line' for _ in names]):
                # if we have four straight lines, we may classify the elements into groups.
                raise NotImplementedError()
            else:
                self._mtype = None

        return self._mtype
