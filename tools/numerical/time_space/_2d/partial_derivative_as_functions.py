# -*- coding: utf-8 -*-

import sys
if './' not in sys.path: 
    sys.path.append('./')

from types import FunctionType, MethodType
from tools.frozen import Frozen

from tools.numerical.time_space._2d.partial_derivative import NumericalPartialDerivativeTxy


class NumericalPartialDerivativeTxyFunctions(Frozen):
    """
    Like the NumericalPartialDerivative_txy class but this will produce (through __call__ method) callable
    functions (method).
    """
    def __init__(self, func):
        self.___PRIVATE_check_func___(func)
        self._freeze()

    def ___PRIVATE_check_func___(self, func):
        assert callable(func), " <PartialDerivative> : func is not callable."
        if isinstance(func, FunctionType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 3, " <PartialDerivative> : need a func of 3 args."
        elif isinstance(func, MethodType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 4, " <PartialDerivative> : need a method of 4 args (4 including self)."
        elif callable(func):
            pass
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
        else:
            raise Exception(f"partial_{p_} is wrong, should be t, x or y.")

    def ___PRIVATE_partial_func___partial_t___(self, t, x, y):
        """pf/dt
        This partial derivative function will only accept input t is a number, i.e. int or float.
        """
        NPD4 = NumericalPartialDerivativeTxy(self._func_, t, x, y)
        return NPD4.partial('t')

    def ___PRIVATE_partial_func___partial_x___(self, t, x, y):
        """pf/dx
        This partial derivative function will only accept input t is a number, i.e. int or float.
        """
        NPD4 = NumericalPartialDerivativeTxy(self._func_, t, x, y)
        return NPD4.partial('x')

    def ___PRIVATE_partial_func___partial_y___(self, t, x, y):
        """pf/dy
        This partial derivative function will only accept input t is a number, i.e. int or float.
        """
        NPD4 = NumericalPartialDerivativeTxy(self._func_, t, x, y)
        return NPD4.partial('y')


if __name__ == '__main__':
    # mpiexec -n 6 python tools/numerical/time_space/_2d/partial_derivative_as_functions.py
    import numpy as np

    def func(t, x, y): return np.sin(np.pi*x) * np.sin(np.pi*y) * t

    def Pt(t, x, y): return np.sin(np.pi*x) * np.sin(np.pi*y) + 0*t
    def Px(t, x, y): return np.pi*np.cos(np.pi*x) * np.sin(np.pi*y) * t
    def Py(t, x, y): return np.pi*np.sin(np.pi*x) * np.cos(np.pi*y) * t

    t = 5
    x = np.random.rand(11, 12)
    y = np.random.rand(11, 12)

    NP = NumericalPartialDerivativeTxy(func, t, x, y)
    #
    assert all(NP.check_total(Pt, Px, Py))

    NPD4F = NumericalPartialDerivativeTxyFunctions(func)

    Npt = NPD4F('t')
    Npx = NPD4F('x')
    Npy = NPD4F('y')

    # noinspection PyArgumentList
    print(np.sum(np.abs(Pt(t, x, y) - Npt(t, x, y))))
    # noinspection PyArgumentList
    print(np.sum(np.abs(Px(t, x, y) - Npx(t, x, y))))
    # noinspection PyArgumentList
    print(np.sum(np.abs(Py(t, x, y) - Npy(t, x, y))))
