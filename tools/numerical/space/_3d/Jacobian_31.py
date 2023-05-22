# -*- coding: utf-8 -*-
from abc import ABC
import numpy as np
from tools.numerical.derivative import derivative


class NumericalJacobianXYZt31(ABC):
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

    def total_derivative(self, t, h=1e-6):
        Xt = derivative(self.___evaluate_func31_for_x_t___, t, h=h)
        Yt = derivative(self.___evaluate_func31_for_y_t___, t, h=h)
        Zt = derivative(self.___evaluate_func31_for_z_t___, t, h=h)
        return Xt, Yt, Zt

    def check_Jacobian(self, Jacobian, t, tolerance=1e-6):
        """Check if ``Jacobian(t) == self.scipy_derivative(t)`` at nodes ``t``."""
        self_J = self.total_derivative(t)
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
