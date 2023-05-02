# -*- coding: utf-8 -*-
from abc import ABC
from tools.numerical.space._2d.partial_derivative import NumericalPartialDerivative_xy


class NumericalJacobian_xy_22(ABC):
    """
    For a mapping: ``x = Phi_x(r, s), y = Phi_y(r, s)``,
    ``self._func_(r, s) = (Phi_x(r, s), Phi_y(r, s))``, we compute its Jacobian numerically:
    ``(( dx/dr, dx/ds ), ( dy/dr, dy/ds ))``.

    """
    def __init__(self, func22):
        """ """
        self._func22_ = func22

    def ___PRIVATE_evaluate_func22_for_x_rs___(self, r, s):
        return self._func22_(r, s)[0]

    def ___PRIVATE_evaluate_func22_for_y_rs___(self, r, s):
        return self._func22_(r, s)[1]

    def scipy_derivative(self, r, s, dr_ds=1e-8, n=1, order=3):
        xr, xs = NumericalPartialDerivative_xy(self.___PRIVATE_evaluate_func22_for_x_rs___,
                                               r, s, dx_dy=dr_ds, n=n, order=order).scipy_total
        yr, ys = NumericalPartialDerivative_xy(self.___PRIVATE_evaluate_func22_for_y_rs___,
                                               r, s, dx_dy=dr_ds, n=n, order=order).scipy_total
        return ((xr, xs),
                (yr, ys))
