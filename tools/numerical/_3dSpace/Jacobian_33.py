# -*- coding: utf-8 -*-
"""3D numerical."""
from abc import ABC
from tools.numerical._3dSpace.partial_derivative import NumericalPartialDerivative_xyz


class NumericalJacobian_xyz_33(ABC):
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

    def ___evaluate_func33_for_x_rst___(self, r, s, t):
        return self._func33_(r, s, t)[0]

    def ___evaluate_func33_for_y_rst___(self, r, s, t):
        return self._func33_(r, s, t)[1]

    def ___evaluate_func33_for_z_rst___(self, r, s, t):
        return self._func33_(r, s, t)[2]

    def scipy_derivative(self, r, s, t, drdsdt=1e-8, n=1, order=3):
        xr, xs, xt = NumericalPartialDerivative_xyz(self.___evaluate_func33_for_x_rst___,
                                                    r, s, t, dxdydz=drdsdt, n=n, order=order).scipy_total
        yr, ys, yt = NumericalPartialDerivative_xyz(self.___evaluate_func33_for_y_rst___,
                                                    r, s, t, dxdydz=drdsdt, n=n, order=order).scipy_total
        zr, zs, zt = NumericalPartialDerivative_xyz(self.___evaluate_func33_for_z_rst___,
                                                    r, s, t, dxdydz=drdsdt, n=n, order=order).scipy_total
        return ((xr, xs, xt),
                (yr, ys, yt),
                (zr, zs, zt))
