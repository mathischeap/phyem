# -*- coding: utf-8 -*-
"""
"""
from phyem.tools.frozen import Frozen


class MsePyRootFormInterpolateCopy(Frozen):
    """"""

    def __init__(self, rf, t):
        """"""
        assert rf._is_base()  # we make copy only from a base form.
        self._f = rf
        self._t = t
        self._freeze()
