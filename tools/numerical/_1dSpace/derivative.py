# -*- coding: utf-8 -*-
"""1D numerical."""
from types import FunctionType, MethodType
import numpy as np
from tools.frozen import Frozen
from scipy.misc import derivative


class NumericalDerivative_fx(Frozen):
    """Numerical derivative in 1D."""
    def __init__(self, func, x, dx=1e-6, n=1, order=3):
        """
        Parameters
        ----------
        dx :
            The interval. The smaller, more accurate.
        n :
            `n`th order derivative.
        order:
            How many points are used to approximate the derivative.

        """
        self.___PRIVATE_check_func___(func)
        self.___PRIVATE_check_x___(x)
        self.___PRIVATE_check_dx___(dx)
        self.___PRIVATE_check_n___(n)
        self.___PRIVATE_check_order___(order)
        self._freeze()

    def ___PRIVATE_check_func___(self, func):
        assert callable(func), " <PartialDerivative> : func is not callable."
        if isinstance(func, FunctionType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 1, " <PartialDerivative> : need a func of 1 args."
        elif isinstance(func, MethodType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 2, \
                " <PartialDerivative> : need a method of 1 args (2 including self)."
        else:
            raise NotImplementedError()
        self._func_ = func

    def ___PRIVATE_check_x___(self, x):
        self._x_ = x
        assert np.ndim(self._x_) == 1

    def ___PRIVATE_check_dx___(self, dx):
        """ """
        assert isinstance(dx, (int, float))
        self._dx_ = dx

    def ___PRIVATE_check_n___(self, n):
        """ """
        assert n % 1 == 0 and n >= 1, " <PartialDerivative> : n = {} is wrong.".format(n)
        self._n_ = n

    def ___PRIVATE_check_order___(self, order):
        """ """
        assert order % 2 == 1 and order > 0, " <PartialDerivative> : order needs to be odd positive."
        self._order_ = order

    def scipy_derivative(self):
        """We compute ``df/d_`` at points ``x``."""
        # noinspection PyTypeChecker
        return derivative(self._func_, self._x_, dx=self._dx_, n=self._n_, order=self._order_)

    def check_derivative(self, d_func, tolerance=1e-5):
        """
        Given a ``d_func``, this method check if the function is derivative of
        ``self._func_``.
        """
        self_derivative = self.scipy_derivative()
        func_derivative = d_func(self._x_)
        absolute_error = np.max(np.abs(func_derivative-self_derivative))
        if absolute_error < tolerance:
            return True
        relative_error = np.max(np.abs((func_derivative-self_derivative)/self_derivative))
        if relative_error < tolerance:
            return True
        else:
            return False
