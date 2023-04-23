# -*- coding: utf-8 -*-
import sys
if './' not in sys.path:
    sys.path.append('./')

from abc import ABC
from scipy.misc import derivative
import numpy as np
from types import FunctionType, MethodType


class NumericalPartialDerivative_txyz(ABC):
    """
    Numerical partial derivative, we call it '4' because we compute a function or method of 4 inputs like
    ``A=f(t,x,y,z)``. And we will evaluate dA/dt, dA/dx, dA/dy, dA/dz at `(t, x, y, z)`. Note that `(x,y,z)`
    must be of the same shape; no matter the dimensions (we do not do mesh grid to them). And t must be 1-d.
    """
    def __init__(self, func, t, x, y, z, dtdxdydz=1e-6, n=1, order=3):
        self.___PRIVATE_check_func___(func)
        self.___PRIVATE_check_txyz___(t, x, y, z)
        self.___PRIVATE_check_dtdxdydz___(dtdxdydz)
        self.___PRIVATE_check_n___(n)
        self.___PRIVATE_check_order___(order)

    def ___PRIVATE_check_func___(self, func):
        assert callable(func), " <PartialDerivative> : func is not callable."
        if isinstance(func, FunctionType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 4, " <PartialDerivative> : need a func of 4 args."
        elif isinstance(func, MethodType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 5, \
                " <PartialDerivative> : need a method of 5 args (5 including self)."
        else:
            raise NotImplementedError(func.__class__.__name__)
        self._func_ = func

    def ___PRIVATE_check_txyz___(self, t, x, y, z):
        """We ask x, y, z, must be of the same shape."""
        assert np.shape(x) == np.shape(y) == np.shape(z), " <PartialDerivative> : xyz of different shapes."
        self._x_, self._y_, self._z_ = x, y, z
        assert isinstance(t, (int, float)), f"t need to be a number, now t={t} is a {t.__class__}."
        self._t_ = t

    def ___PRIVATE_check_dtdxdydz___(self, dtdxdydz):
        if isinstance(dtdxdydz, (int, float)):
            self._dt_ = self._dx_ = self._dy_ = self._dz_ = dtdxdydz
        else:
            assert np.shape(dtdxdydz) == (4,), " <PartialDerivative> : dtdxdydz shape wrong."
            self._dt_, self._dx_, self._dy_, self._dz_ = dtdxdydz
        assert all([isinstance(d, (int, float)) and d > 0 for d in (self._dt_, self._dx_, self._dy_, self._dz_)]), \
            f"dt, dx, dy, dz must be positive number."

    def ___PRIVATE_check_n___(self, n):
        assert n % 1 == 0 and n >= 1, " <PartialDerivative> : n = {} is wrong.".format(n)
        self._n_ = n

    def ___PRIVATE_check_order___(self, order):
        assert order % 2 == 1 and order > 0, " <PartialDerivative> : order needs to be odd positive."
        self._order_ = order

    def ___PRIVATE_evaluate_func_for_t___(self, t):
        return self._func_(t, self._x_, self._y_, self._z_)

    def ___PRIVATE_evaluate_func_for_x___(self, x):
        return self._func_(self._t_, x, self._y_, self._z_)

    def ___PRIVATE_evaluate_func_for_y___(self, y):
        return self._func_(self._t_, self._x_, y, self._z_)

    def ___PRIVATE_evaluate_func_for_z___(self, z):
        return self._func_(self._t_, self._x_, self._y_, z)

    def scipy_partial(self, d_):
        """We compute the partial derivative, i.e. ``df/d_``, at points ``*txyz``."""
        if d_ == 't':
            # noinspection PyTypeChecker
            return derivative(self.___PRIVATE_evaluate_func_for_t___, self._t_, dx=self._dt_,
                              n=self._n_, order=self._order_)
        elif d_ == 'x':
            # noinspection PyTypeChecker
            return derivative(self.___PRIVATE_evaluate_func_for_x___, self._x_, dx=self._dx_,
                              n=self._n_, order=self._order_)
        elif d_ == 'y':
            # noinspection PyTypeChecker
            return derivative(self.___PRIVATE_evaluate_func_for_y___, self._y_, dx=self._dy_,
                              n=self._n_, order=self._order_)
        elif d_ == 'z':
            # noinspection PyTypeChecker
            return derivative(self.___PRIVATE_evaluate_func_for_z___, self._z_, dx=self._dz_,
                              n=self._n_, order=self._order_)
        else:
            raise Exception(" <PartialDerivative> : dt, dx or dy or dz? give me 't', 'x', 'y' or 'z'.")

    @property
    def scipy_total(self):
        """Use scipy to compute the total derivative."""
        pt = self.scipy_partial('t')
        px = self.scipy_partial('x')
        py = self.scipy_partial('y')
        pz = self.scipy_partial('z')
        return pt, px, py, pz

    def check_partial_t(self, px_func, tolerance=1e-5):
        """give a analytical function `px_func`, we check if it is the partial-t derivative of the self.func"""
        self_pt = self.scipy_partial('t')
        func_pt = px_func(self._t_, self._x_, self._y_, self._z_)
        absolute_error = np.max(np.abs(func_pt-self_pt))
        if absolute_error < tolerance:
            return True
        relative_error = np.max(np.abs((func_pt-self_pt)/self_pt))
        if relative_error < tolerance:
            return True
        else:
            return False

    def check_partial_x(self, px_func, tolerance=1e-5):
        """give a analytical function `px_func`, we check if it is the partial-x derivative of the self.func"""
        self_px = self.scipy_partial('x')
        func_px = px_func(self._t_, self._x_, self._y_, self._z_)
        absolute_error = np.max(np.abs(func_px-self_px))
        if absolute_error < tolerance:
            return True
        relative_error = np.max(np.abs((func_px-self_px)/self_px))
        if relative_error < tolerance:
            return True
        else:
            return False

    def check_partial_y(self, py_func, tolerance=1e-5):
        """give a analytical function `px_func`, we check if it is the partial-y derivative of the self.func"""
        self_py = self.scipy_partial('y')
        func_py = py_func(self._t_, self._x_, self._y_, self._z_)
        absolute_error = np.max(np.abs(func_py-self_py))
        if absolute_error < tolerance:
            return True
        relative_error = np.max(np.abs((func_py-self_py)/self_py))
        if relative_error < tolerance:
            return True
        else:
            return False

    def check_partial_z(self, pz_func, tolerance=1e-5):
        """give a analytical function `px_func`, we check if it is the partial-z derivative of the self.func"""
        self_pz = self.scipy_partial('z')
        func_pz = pz_func(self._t_, self._x_, self._y_, self._z_)
        absolute_error = np.max(np.abs(func_pz-self_pz))
        if absolute_error < tolerance:
            return True
        relative_error = np.max(np.abs((func_pz-self_pz)/func_pz))
        if relative_error < tolerance:
            return True
        else:
            return False

    def check_total(self, pt_func, px_func, py_func, pz_func, tolerance=1e-5):
        """give four analytical functions, we check if it is the partial-t, -x, -y, -z derivatives of the self.func"""
        return (self.check_partial_t(pt_func, tolerance=tolerance),
                self.check_partial_x(px_func, tolerance=tolerance),
                self.check_partial_y(py_func, tolerance=tolerance),
                self.check_partial_z(pz_func, tolerance=tolerance))


if __name__ == '__main__':
    # mpiexec -n 6 python components\numerical\time_plus_3d_space\partial_derivative.py

    def func(t, x, y, z): return np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z) * t

    def Pt(t, x, y, z): return np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z) + 0*t
    def Px(t, x, y, z): return np.pi*np.cos(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z) * t
    def Py(t, x, y, z): return np.pi*np.sin(np.pi*x) * np.cos(np.pi*y) * np.sin(np.pi*z) * t
    def Pz(t, x, y, z): return np.pi*np.sin(np.pi*x) * np.sin(np.pi*y) * np.cos(np.pi*z) * t

    t = 5
    x = np.random.rand(11, 12, 13)
    y = np.random.rand(11, 12, 13)
    z = np.random.rand(11, 12, 13)

    NP = NumericalPartialDerivative_txyz(func, t, x, y, z)
    assert all(NP.check_total(Pt, Px, Py, Pz))
