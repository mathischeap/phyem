# -*- coding: utf-8 -*-
from abc import ABC
import numpy as np
from tools.numerical._1dSpace.derivative import NumericalDerivative_fx


class NumericalJacobian_xyz_t_31(ABC):
    """
    For a mapping: ``XY(t) = (x, y, z) = (X(t), Y(t), Z(t))``, we compute ``dx/dt``, ``dy/dt``, and ``dz/dt``.

    """
    def __init__(self, func31):
        """ """
        self._func31_ = func31

    def ___evaluate_func31_for_x_t___(self, t):
        return self._func31_(t)[0]

    def ___evaluate_func31_for_y_t___(self, t):
        return self._func31_(t)[1]

    def ___evaluate_func31_for_z_t___(self, t):
        return self._func31_(t)[2]

    def scipy_derivative(self, t, dt=1e-6, n=1, order=3):
        Xt = NumericalDerivative_fx(self.___evaluate_func31_for_x_t___, t,
                                    dx=dt, n=n, order=order).scipy_derivative()
        Yt = NumericalDerivative_fx(self.___evaluate_func31_for_y_t___, t,
                                    dx=dt, n=n, order=order).scipy_derivative()
        Zt = NumericalDerivative_fx(self.___evaluate_func31_for_z_t___, t,
                                    dx=dt, n=n, order=order).scipy_derivative()
        return Xt, Yt, Zt

    def check_Jacobian(self, Jacobian, t, tolerance=1e-6):
        """Check if ``Jacobian(t) == self.scipy_derivative(t)`` at nodes ``t``."""
        self_J = self.scipy_derivative(t)
        give_J = Jacobian(t)
        result = [None, None, None]
        for i in range(3):
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
