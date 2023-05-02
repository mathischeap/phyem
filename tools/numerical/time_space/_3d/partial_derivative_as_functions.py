# -*- coding: utf-8 -*-
import sys
if './' not in sys.path:
    sys.path.append('/')

from types import FunctionType, MethodType
import numpy as np
from tools.frozen import Frozen

from tools.numerical.time_space._3d.partial_derivative import NumericalPartialDerivative_txyz


class NumericalPartialDerivative_txyz_Functions(Frozen):
    """Like the NumericalPartialDerivative_txyz class but this will produce (through __call__ method) callable
    functions (method).
    """
    def __init__(self, func):
        self.___PRIVATE_check_func___(func)
        self._freeze()

    def ___PRIVATE_check_func___(self, func):
        assert callable(func), " <PartialDerivative> : func is not callable."
        if isinstance(func, FunctionType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 4, " <PartialDerivative> : need a func of 4 args."
        elif isinstance(func, MethodType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 5, " <PartialDerivative> : need a method of 5 args (5 including self)."
        else:
            raise NotImplementedError(func.__class__.__name__)
        self._func_ = func

    def __call__(self, p_):
        """partial func / partial _? `P_` be 't', 'x', 'y' or 'z'."""
        if p_ == 't':
            return self.___PRIVATE_partial_func___partial_t___
        elif p_ == 'x':
            return self.___PRIVATE_partial_func___partial_x___
        elif p_ == 'y':
            return self.___PRIVATE_partial_func___partial_y___
        elif p_ == 'z':
            return self.___PRIVATE_partial_func___partial_z___
        else:
            raise Exception(f"partial_{p_} is wrong, should be t, x, y or z.")

    def ___PRIVATE_partial_func___partial_t___(self, t, x, y, z):
        """pf/dt
        This partial derivative function will only accept input t is a number, i.e. int or float.
        """
        NPD4 = NumericalPartialDerivative_txyz(self._func_, t, x, y, z)
        return NPD4.scipy_partial('t')

    def ___PRIVATE_partial_func___partial_x___(self, t, x, y, z):
        """pf/dx
        This partial derivative function will only accept input t is a number, i.e. int or float.
        """
        NPD4 = NumericalPartialDerivative_txyz(self._func_, t, x, y, z)
        return NPD4.scipy_partial('x')

    def ___PRIVATE_partial_func___partial_y___(self, t, x, y, z):
        """pf/dy
        This partial derivative function will only accept input t is a number, i.e. int or float.
        """
        NPD4 = NumericalPartialDerivative_txyz(self._func_, t, x, y, z)
        return NPD4.scipy_partial('y')

    def ___PRIVATE_partial_func___partial_z___(self, t, x, y, z):
        """pf/dz
        This partial derivative function will only accept input t is a number, i.e. int or float.
        """
        NPD4 = NumericalPartialDerivative_txyz(self._func_, t, x, y, z)
        return NPD4.scipy_partial('z')


if __name__ == '__main__':
    # mpiexec -n 6 python components\numerical\time_plus_3d_space\partial_derivative_as_functions.py

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

    NPD4F = NumericalPartialDerivative_txyz_Functions(func)

    Npt = NPD4F('t')
    Npx = NPD4F('x')
    Npy = NPD4F('y')
    Npz = NPD4F('z')
