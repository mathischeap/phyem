# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class t3d_3ScalarAdd(Frozen):
    """"""

    def __init__(self, s0, s1, s2):
        """"""
        self._s0_ = s0
        self._s1_ = s1
        self._s2_ = s2
        self._freeze()

    def __call__(self, t, x, y, z):
        return self._s0_(t, x, y, z) + self._s1_(t, x, y, z) + self._s2_(t, x, y, z)
