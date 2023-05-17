# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 2:13 PM on 5/12/2023
"""

from tools.frozen import Frozen
from msepy.tools.matrix.static.assembled import MsePyStaticAssembledMatrix
from msepy.tools.vector.static.assembled import MsePyStaticAssembledVector
from msepy.tools.linear_system.static.assembled.solve import MsePyStaticLinearSystemAssembledSolve


class MsePyStaticLinearSystemAssembled(Frozen):
    """"""

    def __init__(self, static, A, b):
        """"""
        assert isinstance(A, MsePyStaticAssembledMatrix) and isinstance(b, MsePyStaticAssembledVector), \
            f"A or b type wrong."
        self._static = static
        self._A = A
        self._b = b
        self._solve = MsePyStaticLinearSystemAssembledSolve(self)
        self._freeze()

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def solve(self):
        return self._solve
