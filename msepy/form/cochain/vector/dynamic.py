# -*- coding: utf-8 -*-
r"""
"""
from phyem.msepy.tools.vector.dynamic import MsePyDynamicLocalVector


class MsePyRootFormDynamicCochainVector(MsePyDynamicLocalVector):
    """"""

    def __init__(self, rf, dynamic_cochain):
        """"""
        self._f = rf
        super().__init__(dynamic_cochain)
        self._freeze()
