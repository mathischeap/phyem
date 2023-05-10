# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 11:55 AM on 5/9/2023
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix


class MsePyRootFormMatrix(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._freeze()

    @property
    def incidence(self):
        return self._f.cobounday.incidence_matrix  # no cache, make new copy every time.

    @property
    def mass(self):
        gm = self._f.cochain.gathering_matrix
        M = MsePyStaticLocalMatrix(  # make a new copy every single time.
            self._f.space.mass_matrix(self._f.degree),
            gm,
            gm,
        )
        return M


if __name__ == '__main__':
    # python 
    pass
