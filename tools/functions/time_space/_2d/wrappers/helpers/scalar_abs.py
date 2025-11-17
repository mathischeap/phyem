# -*- coding: utf-8 -*-
r"""
"""
from numpy import abs

from phyem.tools.frozen import Frozen


class t2d_ScalarAbs(Frozen):
    """"""

    def __init__(self, s):
        """"""
        self._s_ = s
        self._freeze()

    def __call__(self, t, x, y):
        return abs(self._s_(t, x, y))
