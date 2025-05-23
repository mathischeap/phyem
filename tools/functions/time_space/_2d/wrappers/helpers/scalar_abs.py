# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from numpy import abs


class t2d_ScalarAbs(Frozen):
    """"""

    def __init__(self, s):
        """"""
        self._s_ = s
        self._freeze()

    def __call__(self, t, x, y):
        return abs(self._s_(t, x, y))
