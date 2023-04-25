# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.tools.vector.local import LocalVector


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

    @property
    def vec(self):
        """The vector of dofs (cochains) of the form at time `t`, \vec{f}^{t}."""
        gm = self._f.cochain.gathering_matrix
        if self._t in self._f.cochain:
            local = self._f.cochain[self._t].local
            return LocalVector(local, gm)  # it is a separate object
        else:
            return LocalVector(None, gm)  # it is a separate object.
