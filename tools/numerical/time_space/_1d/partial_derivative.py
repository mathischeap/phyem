# -*- coding: utf-8 -*-
import sys
if './' not in sys.path:
    sys.path.append('/')

from abc import ABC
import numdifftools as nd
# from scipy.misc import derivative
import numpy as np
from types import FunctionType, MethodType


class NumericalPartialDerivative_tx(ABC):
    """
    partial^n.
    """
    def __init__(self, func, t, x, step=None, n=1, order=2):
        self.___PRIVATE_check_func___(func)
        self.___PRIVATE_check_tx___(t, x)
        self._step_ = step
        self._n_ = n
        self._order_ = order

    def ___PRIVATE_check_func___(self, func):
        assert callable(func), " <PartialDerivative> : func is not callable."
        if isinstance(func, FunctionType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 2, " <PartialDerivative> : need a func of 2 args."
        elif isinstance(func, MethodType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 3, \
                " <PartialDerivative> : need a method of 3 args (3 including self)."
        elif callable(func):
            pass
        else:
            raise NotImplementedError(func.__class__.__name__)
        self._func_ = func

    def ___PRIVATE_check_tx___(self, t, x):
        """We ask x, y, z, must be of the same shape."""
        self._x_ = x
        assert isinstance(t, (int, float)), f"t need to be a number, now t={t} is a {t.__class__}."
        self._t_ = t

    def ___PRIVATE_evaluate_func_for_t___(self, t):
        return self._func_(t, self._x_)

    def ___PRIVATE_evaluate_func_for_x___(self, x):
        return self._func_(self._t_, x)

    def scipy_partial(self, d_):
        """We compute the partial derivative, i.e. ``df/d_``, at points ``*tx``."""
        if d_ == 't':
            # noinspection PyTypeChecker
            return nd.Derivative(
                self.___PRIVATE_evaluate_func_for_t___,
                step=self._step_, n=self._n_, order=self._order_
            )(self._t_)
        elif d_ == 'x':
            # noinspection PyTypeChecker
            return nd.Derivative(
                self.___PRIVATE_evaluate_func_for_x___,
                step=self._step_, n=self._n_, order=self._order_
            )(self._x_)
        else:
            raise Exception(" <PartialDerivative> : dt or dx? give me 't' or 'x'.")

    @property
    def scipy_total(self):
        """Use scipy to compute the total derivative."""
        pt = self.scipy_partial('t')
        px = self.scipy_partial('x')
        return pt, px

    def check_partial_t(self, px_func, tolerance=1e-5):
        """give an analytical function `px_func`, we check if it is the partial-t derivative of the self.func"""
        self_pt = self.scipy_partial('t')
        func_pt = px_func(self._t_, self._x_)
        absolute_error = np.max(np.abs(func_pt-self_pt))
        if absolute_error < tolerance:
            return True
        relative_error = np.max(np.abs((func_pt-self_pt)/self_pt))
        if relative_error < tolerance:
            return True
        else:
            return False

    def check_partial_x(self, px_func, tolerance=1e-5):
        """give an analytical function `px_func`, we check if it is the partial-x derivative of the self.func"""
        self_px = self.scipy_partial('x')
        func_px = px_func(self._t_, self._x_)
        absolute_error = np.max(np.abs(func_px-self_px))
        if absolute_error < tolerance:
            return True
        relative_error = np.max(np.abs((func_px-self_px)/self_px))
        if relative_error < tolerance:
            return True
        else:
            return False

    def check_total(self, pt_func, px_func, tolerance=1e-5):
        """give four analytical functions, we check if it is the partial-t, -x, -y derivatives of the self.func"""
        return (self.check_partial_t(pt_func, tolerance=tolerance),
                self.check_partial_x(px_func, tolerance=tolerance))


if __name__ == '__main__':
    # python .\tools\numerical\_1d\partial_derivative.py
    def func(t, x): return np.sin(np.pi*x) + t

    def Pt(t, x): return x * t * 0 + 1
    def Px(t, x): return np.pi*np.cos(np.pi*x) + 0 * t

    t = 5
    x = np.random.rand(11, 12)

    NP = NumericalPartialDerivative_tx(func, t, x)

    assert all(NP.check_total(Pt, Px))
    print(111)