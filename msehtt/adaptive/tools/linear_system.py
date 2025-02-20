# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHtt_Adaptive_LinearSystem(Frozen):
    """"""

    def __init__(self, obj, top_base):
        """"""
        self._obj = obj
        self._top_base = top_base
        self._freeze()
