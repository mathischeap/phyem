# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""
from msepy.tools.vector.static.local import MsePyStaticLocalVector


class MsePyRootFormStaticCochainVector(MsePyStaticLocalVector):
    """"""

    def __init__(self, rf, t, _2d_data, gathering_matrix):
        self._f = rf
        self._time = t
        super().__init__(_2d_data, gathering_matrix)
        self._freeze()

    def override(self):
        """override `self._data` to be the cochain of `self._f` at time `self._t`."""
        assert self.data is not None, f"I have no data."
        self._f[self._time].cochain = self.data
