# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 4:17 PM on 5/16/2023
"""
from tools.frozen import Frozen
from msepy.tools.vector.static.assembled import MsePyStaticAssembledVector
from numpy import zeros


class MsePyStaticLocalVectorAssemble(Frozen):
    """"""

    def __init__(self, v):
        """"""
        self._v = v
        self._freeze()

    def __call__(self):
        """"""
        gm = self._v._gm
        v = zeros(gm.num_dofs)

        for i in self._v:
            Vi = self._v[i]  # all adjustments and customizations take effect.
            v[gm[i]] += Vi  # must do this to be consistent with the matrix assembling.

        return MsePyStaticAssembledVector(v, gm)
