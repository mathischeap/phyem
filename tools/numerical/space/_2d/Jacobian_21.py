# -*- coding: utf-8 -*-
"""2D numerical."""
import numpy as np
from abc import ABC
from tools.numerical.space._1d.derivative import NumericalDerivative_fx
from tools.numerical.space._2d.Jacobian_22 import NumericalPartialDerivative_xy, NumericalJacobian_xy_22


class NumericalJacobian_xy_t_21(ABC):
    """For a mapping: ``XY(t) = (x, y) = (X(t), Y(t))``, We compute ``dx/dt``, and ``dy/dt``.
    """
    def __init__(self, func21):
        """ """
        self._func21_ = func21

    def ___PRIVATE_evaluate_func21_for_x_t___(self, t):
        return self._func21_(t)[0]

    def ___PRIVATE_evaluate_func21_for_y_t___(self, t):
        return self._func21_(t)[1]

    def scipy_derivative(self, t, dt=1e-6, n=1, order=3):
        Xt = NumericalDerivative_fx(self.___PRIVATE_evaluate_func21_for_x_t___, t,
                                    dx=dt, n=n, order=order).scipy_derivative()
        Yt = NumericalDerivative_fx(self.___PRIVATE_evaluate_func21_for_y_t___, t,
                                    dx=dt, n=n, order=order).scipy_derivative()
        return Xt, Yt

    def check_Jacobian(self, Jacobian, t, tolerance=1e-6):
        """Check if ``Jacobian(t) == self.scipy_derivative(t)`` at nodes ``t``. """
        self_J = self.scipy_derivative(t)
        give_J = Jacobian(t)
        result = [None, None]
        for i in range(2):
            absolute_error = np.max(np.abs(self_J[i]-give_J[i]))
            if absolute_error < tolerance:
                result[i] = True
            else:
                relative_error = np.max(np.abs((self_J[i]-give_J[i])/self_J[i]))
                if relative_error < tolerance:
                    result[i] = True
                else:
                    result[i] = False
        return tuple(result)


if __name__ == '__main__':
    NP = NumericalPartialDerivative_xy
    NJ = NumericalJacobian_xy_22
