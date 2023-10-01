# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.tools.vector.static.assembled import IrregularStaticAssembledVector
from numpy import zeros


class IrregularStaticLocalVectorAssemble(Frozen):
    """"""

    def __init__(self, v):
        """"""
        self._v = v
        self._freeze()

    def __call__(self):
        """"""
        gm = self._v._gm
        v = zeros(gm.num_dofs)

        for i in gm:

            Vi = self._v[i]  # all adjustments and customizations take effect.
            v[gm[i]] += Vi  # must do this to be consistent with the matrix assembling.

        return IrregularStaticAssembledVector(v, gm)
