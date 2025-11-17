# -*- coding: utf-8 -*-
r"""2D numerical."""
import numpy as np
from abc import ABC

from phyem.tools.numerical.derivative import derivative


class NumericalJacobianXYt21(ABC):
    """For a mapping: ``XY(t) = (x, y) = (X(t), Y(t))``, We compute ``dx/dt``, and ``dy/dt``.
    """
    def __init__(self, func21):
        """ """
        self._func21_ = func21

    def _evaluate_func21_for_x_t(self, t):
        return self._func21_(t)[0]

    def _evaluate_func21_for_y_t(self, t):
        return self._func21_(t)[1]

    def total_derivative(self, t, h=1e-6):
        Xt = derivative(self._evaluate_func21_for_x_t, t, h=h)
        Yt = derivative(self._evaluate_func21_for_y_t, t, h=h)
        return Xt, Yt

    def check_Jacobian(self, Jacobian, t, tolerance=1e-6):
        """Check if ``Jacobian(t) == self.scipy_derivative(t)`` at nodes ``t``. """
        self_J = self.total_derivative(t)
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
