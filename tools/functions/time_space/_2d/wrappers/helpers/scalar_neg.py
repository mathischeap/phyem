# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class t2d_ScalarNeg(Frozen):
    """"""

    def __init__(self, s):
        """"""
        self._s_ = s
        self._freeze()

    def __call__(self, t, x, y):
        return - self._s_(t, x, y)
