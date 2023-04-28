# -*- coding: utf-8 -*-

"""

"""
import sys

from types import FunctionType, MethodType

if './' not in sys.path:
    sys.path.append('./')

__all__ = [
    "vector",
]

from tools.functions.timePlus1dSpace.wrappers.scalar import t1dScalar as _1dt_scalar
from tools.functions.timePlus2dSpace.wrappers.scalar import t2dScalar as _2dt_scalar
from tools.functions.timePlus3dSpace.wrappers.scalar import t3dScalar as _3dt_scalar
from tools.functions.timePlus2dSpace.wrappers.vector import t2dVector as _2dt_vector
from tools.functions.timePlus3dSpace.wrappers.vector import t3dVector as _3dt_vector
from tools.functions.timePlus2dSpace.wrappers.tensor import t2dTensor as _2dt_tensor
from tools.functions.timePlus3dSpace.wrappers.tensor import t3dTensor as _3dt_tensor


def scalar(func):
    """"""
    assert callable(func), " <PartialDerivative> : func is not callable."
    if isinstance(func, FunctionType):
        # noinspection PyUnresolvedReferences
        num_arg = func.__code__.co_argcount
        n = num_arg
    elif isinstance(func, MethodType):
        # noinspection PyUnresolvedReferences
        num_arg = func.__code__.co_argcount
        n = num_arg - 1
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

    if len(funcs) == 2:
        return _2dt_tensor(*funcs)
    elif len(funcs) == 3:
        return _3dt_tensor(*funcs)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    # python tools/vector_calculus.py
    print(vector)
