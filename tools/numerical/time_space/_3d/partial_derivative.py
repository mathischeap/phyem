# -*- coding: utf-8 -*-
import sys
if './' not in sys.path:
    sys.path.append('./')

from abc import ABC
import numpy as np
from types import FunctionType, MethodType
from tools.numerical.derivative import derivative


class NumericalPartialDerivativeTxyz(ABC):
    """Numerical partial derivative, we call it '4' because we compute a function or method of 4 inputs like
    ``A=f(t,x,y,z)``. And we will evaluate dA/dt, dA/dx, dA/dy, dA/dz at `(t, x, y, z)`. Note that `(x,y,z)`
    must be of the same shape; no matter the dimensions (we do not do mesh grid to them). And t must be 1-d.
    """
    def __init__(self, func, t, x, y, z, h=1e-6):
        self._check_func(func)
        self._check_txyz(t, x, y, z)
        self._h = h

    def _check_func(self, func):
        assert callable(func), " <PartialDerivative> : func is not callable."
        if isinstance(func, FunctionType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 4, " <PartialDerivative> : need a func of 4 args."
        elif isinstance(func, MethodType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 5, \
                " <PartialDerivative> : need a method of 5 args (5 including self)."
        elif callable(func):
            pass
        else:
            raise NotImplementedError(func.__class__.__name__)
        self._func_ = func

    def _check_txyz(self, t, x, y, z):
        """We ask x, y, z, must be of the same shape."""
        assert np.shape(x) == np.shape(y) == np.shape(z), " <PartialDerivative> : xyz of different shapes."
        self._x_, self._y_, self._z_ = x, y, z
        assert isinstance(t, (int, float)), f"t need to be a number, now t={t} is a {t.__class__}."
        self._t_ = t

    def _evaluate_func_for_t(self, t):
        return self._func_(t, self._x_, self._y_, self._z_)

    def _evaluate_func_for_x(self, x):
        return self._func_(self._t_, x, self._y_, self._z_)

    def _evaluate_func_for_y(self, y):
        return self._func_(self._t_, self._x_, y, self._z_)

    def _evaluate_func_for_z(self, z):
        return self._func_(self._t_, self._x_, self._y_, z)

    def partial(self, d_):
        """We compute the partial derivative, i.e. ``df/d_``, at points ``*txyz``."""
        if d_ == 't':
            data = derivative(self._evaluate_func_for_t, self._t_, h=self._h)
        elif d_ == 'x':
            data = derivative(self._evaluate_func_for_x, self._x_, h=self._h)
        elif d_ == 'y':
            data = derivative(self._evaluate_func_for_y, self._y_, h=self._h)
        elif d_ == 'z':
            data = derivative(self._evaluate_func_for_z, self._z_, h=self._h)
        else:
            raise Exception(" <PartialDerivative> : dt, dx or dy or dz? give me 't', 'x', 'y' or 'z'.")

        return data

    @property
    def total_partial(self):
        """Use scipy to compute the total derivative."""
        pt = self.partial('t')
        px = self.partial('x')
        py = self.partial('y')
        pz = self.partial('z')
        return pt, px, py, pz

    def check_partial_t(self, px_func, tolerance=1e-5):
        """give a analytical function `px_func`, we check if it is the partial-t derivative of the self.func"""
        self_pt = self.partial('t')
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
        self_px = self.partial('x')
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
        self_py = self.partial('y')
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
        self_pz = self.partial('z')
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
    # mpiexec -n 6 python tools/numerical/time_space/_3d/partial_derivative.py

    def func(t, x, y, z): return np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z) * t

    def Pt(t, x, y, z): return np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z) + 0*t
    def Px(t, x, y, z): return np.pi*np.cos(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z) * t
    def Py(t, x, y, z): return np.pi*np.sin(np.pi*x) * np.cos(np.pi*y) * np.sin(np.pi*z) * t
    def Pz(t, x, y, z): return np.pi*np.sin(np.pi*x) * np.sin(np.pi*y) * np.cos(np.pi*z) * t

    t = 5
    x = np.random.rand(11, 12, 13)
    y = np.random.rand(11, 12, 13)
    z = np.random.rand(11, 12, 13)

    NP = NumericalPartialDerivativeTxyz(func, t, x, y, z)
    assert all(NP.check_total(Pt, Px, Py, Pz))
