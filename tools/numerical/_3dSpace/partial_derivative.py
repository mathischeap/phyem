# -*- coding: utf-8 -*-

from types import FunctionType, MethodType
from abc import ABC
from scipy.misc import derivative
import numpy as np


class NumericalPartialDerivative_xyz(ABC):
    """
    Numerical partial derivative, we call it '3' because we compute a function or method that like:
    ``a=f(x,y,z)``.
    """
    def __init__(self, func, x, y, z, dxdydz=1e-6, n=1, order=3):
        self.___check_func___(func)
        self.___check_xyz___(x, y, z)
        self.___check_n___(n)
        self.___check_order___(order)
        self.___check_dxdydz___(dxdydz)

    def ___check_func___(self, func):
        assert callable(func), " <PartialDerivative> : func is not callable."
        if isinstance(func, FunctionType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 3, " <PartialDerivative> : need a func of 3 args."
        elif isinstance(func, MethodType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 4, \
                " <PartialDerivative> : need a method of 3 args (4 including self)."
        elif func.__class__.__name__ == 'partial':
            # noinspection PyUnresolvedReferences
            if isinstance(func.func, FunctionType):
                # noinspection PyUnresolvedReferences
                assert func.func.__code__.co_argcount == 4
            elif isinstance(func.func, MethodType):
                # noinspection PyUnresolvedReferences
                assert func.func.__code__.co_argcount == 5
            else:
                raise Exception()
        else:
            raise NotImplementedError(func.__class__.__name__)
        self._func_ = func

    def ___check_xyz___(self, x, y, z):
        self._x_, self._y_, self._z_ = x, y, z
        assert np.shape(self._x_) == np.shape(self._y_) == np.shape(self._z_), \
            " <PartialDerivative> : xyz of different shapes."

    def ___check_dxdydz___(self, dxdydz):
        if isinstance(dxdydz, (int, float)):
            self._dx_ = self._dy_ = self._dz_ = dxdydz
        else:
            assert np.shape(dxdydz) == (3,), " <PartialDerivative> : dxdydz shape wrong."
            self._dx_, self._dy_, self._dz_ = dxdydz
        assert all([isinstance(d, (int, float)) and d > 0 for d in (self._dx_, self._dy_, self._dz_)])

    def ___check_n___(self, n):
        assert n % 1 == 0 and n >= 1, " <PartialDerivative> : n = {} is wrong.".format(n)
        self._n_ = n

    def ___check_order___(self, order):
        assert order % 2 == 1 and order > 0, " <PartialDerivative> : order needs to be odd positive."
        self._order_ = order

    def ___evaluate_func_for_x___(self, x):
        return self._func_(x, self._y_, self._z_)

    def ___evaluate_func_for_y___(self, y):
        return self._func_(self._x_, y, self._z_)

    def ___evaluate_func_for_z___(self, z):
        return self._func_(self._x_, self._y_, z)

    def scipy_partial(self, d_):
        """We compute ``df/d_`` at points ``*xyz``."""
        if d_ == 'x':
            # noinspection PyTypeChecker
            return derivative(self.___evaluate_func_for_x___, self._x_, dx=self._dx_,
                              n=self._n_, order=self._order_)
        elif d_ == 'y':
            # noinspection PyTypeChecker
            return derivative(self.___evaluate_func_for_y___, self._y_, dx=self._dy_,
                              n=self._n_, order=self._order_)
        elif d_ == 'z':
            # noinspection PyTypeChecker
            return derivative(self.___evaluate_func_for_z___, self._z_, dx=self._dz_,
                              n=self._n_, order=self._order_)
        else:
            raise Exception(" <PartialDerivative> : dx or dy or dz? ")

    @property
    def scipy_total(self):
        px = self.scipy_partial('x')
        py = self.scipy_partial('y')
        pz = self.scipy_partial('z')
        return px, py, pz

    def check_partial_x(self, px_func, tolerance=1e-5):
        self_px = self.scipy_partial('x')
        func_px = px_func(self._x_, self._y_, self._z_)
        absolute_error = np.max(np.abs(func_px-self_px))
        if absolute_error < tolerance:
            return True
        relative_error = np.max(np.abs((func_px-self_px)/self_px))
        if relative_error < tolerance:
            return True
        else:
            return False

    def check_partial_y(self, py_func, tolerance=1e-5):
        self_py = self.scipy_partial('y')
        func_py = py_func(self._x_, self._y_, self._z_)
        absolute_error = np.max(np.abs(func_py-self_py))
        if absolute_error < tolerance:
            return True
        relative_error = np.max(np.abs((func_py-self_py)/self_py))
        if relative_error < tolerance:
            return True
        else:
            return False

    def check_partial_z(self, pz_func, tolerance=1e-5):
        self_pz = self.scipy_partial('z')
        func_pz = pz_func(self._x_, self._y_, self._z_)
        absolute_error = np.max(np.abs(func_pz-self_pz))
        if absolute_error < tolerance:
            return True
        relative_error = np.max(np.abs((func_pz-self_pz)/func_pz))
        if relative_error < tolerance:
            return True
        else:
            return False

    def check_total(self, px_func, py_func, pz_func, tolerance=1e-5):
        return (self.check_partial_x(px_func, tolerance=tolerance),
                self.check_partial_y(py_func, tolerance=tolerance),
                self.check_partial_z(pz_func, tolerance=tolerance))
