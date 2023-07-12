# -*- coding: utf-8 -*-

import sys
if './' not in sys.path: 
    sys.path.append('./')

from types import FunctionType, MethodType
from tools.frozen import Frozen

from tools.numerical.time_space._1d.partial_derivative import NumericalPartialDerivativeTx


class NumericalPartialDerivativeTxFunctions(Frozen):
    """
    Like the NumericalPartialDerivative_txy class but this will produce (through __call__ method) callable
    functions (method).
    """
    def __init__(self, func):
        self._check_func(func)
        self._freeze()

    def _check_func(self, func):
        assert callable(func), " <PartialDerivative> : func is not callable."
        if isinstance(func, FunctionType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 2, " <PartialDerivative> : need a func of 2 args."
        elif isinstance(func, MethodType):
            # noinspection PyUnresolvedReferences
            assert func.__code__.co_argcount == 3, " <PartialDerivative> : need a method of 3 args (3 including self)."
        elif callable(func):
            pass
        else:
            raise NotImplementedError(func.__class__.__name__)
        self._func_ = func

    def __call__(self, p_):
        """partial func / partial _? `P_` be 't' or 'x'."""
        if p_ == 't':
            return self._partial_func___partial_t
        elif p_ == 'x':
            return self._partial_func___partial_x
        else:
            raise Exception(f"partial_{p_} is wrong, should be t, or x.")

    def _partial_func___partial_t(self, t, x):
        """pf/dt
        This partial derivative function will only accept input t is a number, i.e. int or float.
        """
        NPD4 = NumericalPartialDerivativeTx(self._func_, t, x)
        return NPD4.partial('t')

    def _partial_func___partial_x(self, t, x):
        """pf/dx
        This partial derivative function will only accept input t is a number, i.e. int or float.
        """
        NPD4 = NumericalPartialDerivativeTx(self._func_, t, x)
        return NPD4.partial('x')


if __name__ == '__main__':
    # python tools/numerical/time_space/_1d/partial_derivative_as_functions.py
    import numpy as np

    def func(t, x): return np.sin(np.pi*x) * t

    def Pt(t, x): return np.sin(np.pi*x) + 0*t
    def Px(t, x): return np.pi*np.cos(np.pi*x) * t

    t = 5
    x = np.random.rand(11, 12)

    NP = NumericalPartialDerivativeTx(func, t, x)
    #
    assert all(NP.check_total(Pt, Px))

    NPD4F = NumericalPartialDerivativeTxFunctions(func)

    Npt = NPD4F('t')
    Npx = NPD4F('x')

    # noinspection PyArgumentList
    print(np.sum(np.abs(Pt(t, x) - Npt(t, x))))
    # noinspection PyArgumentList
    print(np.sum(np.abs(Px(t, x) - Npx(t, x))))
