# -*- coding: utf-8 -*-

"""

"""
import sys

if './' not in sys.path:
    sys.path.append('./')

__all__ = [
    "vector",
]


from tools.functions.timePlus2dSpace.wrappers.scalar import t2dScalar as _2dt_scalar
from tools.functions.timePlus3dSpace.wrappers.scalar import t3dScalar as _3dt_scalar
from tools.functions.timePlus2dSpace.wrappers.vector import t2dVector as _2dt_vector
from tools.functions.timePlus3dSpace.wrappers.vector import t3dVector as _3dt_vector
from tools.functions.timePlus2dSpace.wrappers.tensor import t2dTensor as _2dt_tensor
from tools.functions.timePlus3dSpace.wrappers.tensor import t3dTensor as _3dt_tensor


def scalar(*funcs):
    """"""

    if len(funcs) == 2:
        return _2dt_scalar(*funcs)
    elif len(funcs) == 3:
        return _3dt_scalar(*funcs)
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
