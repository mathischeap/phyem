# -*- coding: utf-8 -*-

"""

"""
import sys

if './' not in sys.path:
    sys.path.append('./')

__all__ = [
    "vector",
]


from tools.functions.timePlus2dSpace.wrappers.vector import t2dVector as _2dt_vector

def vector(*funcs):
    """"""

    if len(funcs) == 2:
        return _2dt_vector(*funcs)


if __name__ == '__main__':
    # python tools/vector_calculus.py
    print(vector)