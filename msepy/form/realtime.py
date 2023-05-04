# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.tools.vector.local import MsePyLocalVector


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
        """The vector of dofs (cochain) of the form at time `t`, \vec{f}^{t}."""
        gm = self._f.cochain.gathering_matrix
        if self._t in self._f.cochain:
            local = self._f.cochain[self._t].local
            return MsePyLocalVector(local, gm)  # it is a separate object
        else:
            return MsePyLocalVector(None, gm)  # it is a separate object.

    def reduce(self, update_cochain=True, **kwargs):
        self._f.reduce(self._t, update_cochain=update_cochain, **kwargs)

    def reconstruct(self, *meshgrid, **kwargs):
        """"""
        return self._f.reconstruct(self._t, *meshgrid, **kwargs)

    @property
    def visualize(self):
        """"""
        return self._f.visualize[self._t]

    @property
    def error(self):
        """"""
        return self._f.error[self._t]

    @property
    def coboundary(self):
        """"""
        return self._f.coboundary[self._t]
