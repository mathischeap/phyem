# -*- coding: utf-8 -*-
r"""
"""
from msehtt.tools.vector.dynamic import MseHttDynamicLocalVector


class MseHttDynamicCochainVector(MseHttDynamicLocalVector):
    """"""

    def __init__(self, rf, dynamic_cochain):
        """"""
        self._f = rf
        super().__init__(dynamic_cochain)
        self._freeze()
