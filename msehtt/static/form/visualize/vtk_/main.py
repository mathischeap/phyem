# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen


class MseHttFormVisualizeVtk(Frozen):
    """"""

    def __init__(self, f, t):
        """"""
        self._f = f
        self._t = t
        self._freeze()

    def __call__(self, saveto, ddf=1):
        """"""
