# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen


class MsePyRootFormRealTimeCopy(Frozen):
    """"""

    def __init__(self, rf, t):
        """"""
        assert rf._is_base()
        self._f = rf
        self._t = t
        self._freeze()

    @property
    def cochain(self):
        """"""
        return self._f.cochain[self._t]

    @cochain.setter
    def cochain(self, cochain):
        """"""
        self._f.cochain._set(self._t, cochain)
