# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from msepy.tools.vector.static.local import MsePyStaticLocalVector


class MsePyRootFormStaticCochainVector(MsePyStaticLocalVector):
    """"""

    def __init__(self, rf, t, _2d_data, gathering_matrix):
        super().__init__(_2d_data, gathering_matrix)
        self._f = rf
        self._t = t
        self._freeze()

    def override(self):
        """override `self._data` to be the cochain of `self._f` at time `self._t`."""
        self._f[self._t].cochain = self._data


if __name__ == '__main__':
    # python 
    pass
