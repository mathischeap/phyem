# -*- coding: utf-8 -*-
r"""3D numerical."""
from abc import ABC
from tools.numerical.space._3d.partial_derivative import NumericalPartialDerivativeXYZ


class NumericalJacobianXYZrst33(ABC):
    """
    For a mapping: ``x = Phi_x(r, s, t), y = Phi_y(r, s, t), z = Phi_z(r, s, t),
    (``self._func_(r, s, t) = (Phi_x(r, s, t), Phi_y(r, s, t), Phi_z(r, s, t))``,
    NOT ``self._func_[i](r, s, t) = ...``),
    we compute the its Jacobian numerically: ``(( dx/dr, dx/ds, dx/dt ),
    ( dy/dr, dy/ds, dy/dt ), ( dz/dr, dz/ds, dz/dt ))``.

    """
    def __init__(self, func33):
        """ """
        self._func33_ = func33

    def _evaluate_func33_for_x_rst(self, r, s, t):
        return self._func33_(r, s, t)[0]

    def _evaluate_func33_for_y_rst(self, r, s, t):
        return self._func33_(r, s, t)[1]

    def _evaluate_func33_for_z_rst(self, r, s, t):
        return self._func33_(r, s, t)[2]

    def Jacobian_matrix(self, r, s, t, h=1e-6):
        xr, xs, xt = NumericalPartialDerivativeXYZ(self._evaluate_func33_for_x_rst,
                                                   r, s, t, h=h)
        yr, ys, yt = NumericalPartialDerivativeXYZ(self._evaluate_func33_for_y_rst,
                                                   r, s, t, h=h)
        zr, zs, zt = NumericalPartialDerivativeXYZ(self._evaluate_func33_for_z_rst,
                                                   r, s, t, h=h)
        return ((xr, xs, xt),
                (yr, ys, yt),
                (zr, zs, zt))
