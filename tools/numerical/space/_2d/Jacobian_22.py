# -*- coding: utf-8 -*-
r""""""
from abc import ABC

from phyem.tools.numerical.space._2d.partial_derivative import NumericalPartialDerivativeXY


class NumericalJacobianXYrs22(ABC):
    """
    For a mapping: ``x = Phi_x(r, s), y = Phi_y(r, s)``,
    ``self._func_(r, s) = (Phi_x(r, s), Phi_y(r, s))``, we compute its Jacobian numerically:
    ``(( dx/dr, dx/ds ), ( dy/dr, dy/ds ))``.

    """
    def __init__(self, func22):
        """ """
        self._func22_ = func22

    def _evaluate_func22_for_x_rs(self, r, s):
        return self._func22_(r, s)[0]

    def _evaluate_func22_for_y_rs(self, r, s):
        return self._func22_(r, s)[1]

    def Jacobian_matrix(self, r, s, h=1e-6):
        xr, xs = NumericalPartialDerivativeXY(
            self._evaluate_func22_for_x_rs, r, s, h=h
        ).total_derivative
        yr, ys = NumericalPartialDerivativeXY(
            self._evaluate_func22_for_y_rs, r, s, h=h
        ).total_derivative

        return ((xr, xs),
                (yr, ys))
