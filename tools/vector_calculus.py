# -*- coding: utf-8 -*-
r"""
"""
import sys

from types import FunctionType, MethodType

import numpy as np

if './' not in sys.path:
    sys.path.append('./')


__all__ = [
    "scalar",
    "vector",
    "tensor",

    "m2n2_scalar_on_lines",
    "m2n2_vector_on_lines",
]


from tools.functions.time_space._1d.wrappers.scalar import T1dScalar as _1dt_scalar
from tools.functions.time_space._2d.wrappers.scalar import T2dScalar as _2dt_scalar
from tools.functions.time_space._3d.wrappers.scalar import T3dScalar as _3dt_scalar
from tools.functions.time_space._2d.wrappers.vector import T2dVector as _2dt_vector
from tools.functions.time_space._3d.wrappers.vector import T3dVector as _3dt_vector
from tools.functions.time_space._2d.wrappers.tensor import T2dTensor as _2dt_tensor
from tools.functions.time_space._3d.wrappers.tensor import T3dTensor as _3dt_tensor


from src.config import get_embedding_space_dim


from tools.functions.time_space._2d.wrappers.vector_on_lines import vector_on_lines as m2n2_vector_on_lines
from tools.functions.time_space._2d.wrappers.scalar_on_lines import scalar_on_lines as m2n2_scalar_on_lines


# noinspection PyUnusedLocal
def _2d_zeros(t, x, y):
    return np.zeros_like(x)


# noinspection PyUnusedLocal
def _3d_zeros(t, x, y, z):
    return np.zeros_like(x)


def scalar(func):
    """"""
    assert callable(func) or func == 0, " <PartialDerivative> : func is not callable or 0."
    if isinstance(func, FunctionType):
        # noinspection PyUnresolvedReferences
        num_arg = func.__code__.co_argcount
        n = num_arg - 1  # because we have time
    elif isinstance(func, MethodType):
        # noinspection PyUnresolvedReferences
        num_arg = func.__code__.co_argcount
        n = num_arg - 2  # because we have time and `self`.
    elif func == 0:
        n = get_embedding_space_dim()
        if n == 2:
            func = _2d_zeros
        elif n == 3:
            func = _3d_zeros
        else:
            raise NotImplementedError()

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
