# -*- coding: utf-8 -*-

"""

"""
import sys

from types import FunctionType, MethodType

if './' not in sys.path:
    sys.path.append('./')


__all__ = [
    "scalar",
    "vector",
    "tensor",
]


from tools.functions.time_space._1d.wrappers.scalar import T1dScalar as _1dt_scalar
from tools.functions.time_space._2d.wrappers.scalar import T2dScalar as _2dt_scalar
from tools.functions.time_space._3d.wrappers.scalar import T3dScalar as _3dt_scalar
from tools.functions.time_space._2d.wrappers.vector import T2dVector as _2dt_vector
from tools.functions.time_space._3d.wrappers.vector import T3dVector as _3dt_vector
from tools.functions.time_space._2d.wrappers.tensor import T2dTensor as _2dt_tensor
from tools.functions.time_space._3d.wrappers.tensor import T3dTensor as _3dt_tensor


def scalar(func):
    """"""
    assert callable(func), " <PartialDerivative> : func is not callable."
    if isinstance(func, FunctionType):
        # noinspection PyUnresolvedReferences
        num_arg = func.__code__.co_argcount
        n = num_arg - 1  # because we have time
    elif isinstance(func, MethodType):
        # noinspection PyUnresolvedReferences
        num_arg = func.__code__.co_argcount
        n = num_arg - 2  # because we have time and `self`.
    else:
        raise NotImplementedError(func.__class__.__name__)

    if n == 1:
        return _1dt_scalar(func)
    elif n == 2:
        return _2dt_scalar(func)
    elif n == 3:
        return _3dt_scalar(func)
    else:
        raise NotImplementedError()


def vector(*funcs):
    """"""

    if len(funcs) == 2:
        return _2dt_vector(*funcs)
    elif len(funcs) == 3:
        return _3dt_vector(*funcs)
    else:
        raise NotImplementedError()


def tensor(*funcs):
    """"""

    if len(funcs) == 4:
        return _2dt_tensor(*funcs)
    elif len(funcs) == 9:
        return _3dt_tensor(*funcs)
    else:
        raise NotImplementedError()
