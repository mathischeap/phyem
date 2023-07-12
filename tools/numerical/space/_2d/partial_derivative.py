# -*- coding: utf-8 -*-

from types import FunctionType, MethodType
from abc import ABC
import numpy as np
from tools.numerical.derivative import derivative


class NumericalPartialDerivativeXY(ABC):
    """
    Numerical partial derivative; we call it '2' because we compute a function or method that
    like: ``a=f(x,y)``.

    :param func:
    :param x:
    :param y:
    :param h:
        The interval. The smaller, the more accurate.
    """
    def __init__(self, func, x, y, h=1e-6):
        self._check_func(func)
        self._check_xy(x, y)
        self._h = h

    def _check_func(self, func):
        """ """
        assert callable(func), " <PartialDerivative> : func is not callable."
        if isinstance(func, FunctionType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 2, " <PartialDerivative> : need a func of 2 args."
        elif isinstance(func, MethodType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 3, \
                " <PartialDerivative> : need a method of 2 args (3 including self)."
        elif func.__class__.__name__ == 'partial':
            # noinspection PyUnresolvedReferences
            if isinstance(func.func, FunctionType):
                # noinspection PyUnresolvedReferences
                assert func.func.__code__.co_argcount == 3
            elif isinstance(func.func, MethodType):
                # noinspection PyUnresolvedReferences
                assert func.func.__code__.co_argcount == 4
            else:
                raise Exception()
        else:
            raise NotImplementedError(func.__class__.__name__)
        self._func_ = func

    def _check_xy(self, x, y):
        """ """
        self._x_, self._y_ = x, y
        assert np.shape(self._x_) == np.shape(self._y_), \
            " <PartialDerivative> : xy of different shapes."

    def _evaluate_func_for_x(self, x):
        return self._func_(x, self._y_)

    def _evaluate_func_for_y(self, y):
        return self._func_(self._x_, y)

    def partial(self, d_):
        """ We compute ``df/d_`` at points ``*xyz.``"""
        if d_ == 'x':
            return derivative(self._evaluate_func_for_x, self._x_, h=self._h)
        elif d_ == 'y':
            return derivative(self._evaluate_func_for_y, self._y_, h=self._h)
        else:
            raise Exception(" <PartialDerivative> : dx or dy or dz? ")

    @property
    def total_derivative(self):
        px = self.partial('x')
        py = self.partial('y')
        return px, py

    def check_partial_x(self, px_func, tolerance=1e-5):
        self_px = self.partial('x')
        func_px = px_func(self._x_, self._y_)
        absolute_error = np.max(np.abs(func_px-self_px))
        if absolute_error < tolerance:
            return True
        relative_error = np.max(np.abs((func_px-self_px)/self_px))
        if relative_error < tolerance:
            return True
        else:
            return False

    def check_partial_y(self, py_func, tolerance=1e-5):
        self_py = self.partial('y')
        func_py = py_func(self._x_, self._y_)
        absolute_error = np.max(np.abs(func_py-self_py))
        if absolute_error < tolerance:
            return True
        relative_error = np.max(np.abs((func_py-self_py)/self_py))
        if relative_error < tolerance:
            return True
        else:
            return False

    def check_total(self, px_func, py_func, tolerance=1e-5):
        return (self.check_partial_x(px_func, tolerance=tolerance),
                self.check_partial_y(py_func, tolerance=tolerance))
