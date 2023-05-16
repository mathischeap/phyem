# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 2:13 PM on 5/12/2023
"""
from scipy.sparse import linalg as spspalinalg

from tools.frozen import Frozen
from msepy.tools.matrix.static.assembled import MsePyStaticAssembledMatrix
from msepy.tools.vector.static.assembled import MsePyStaticAssembledVector


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


class MsePyStaticLinearSystemAssembledSolve(Frozen):
    """"""
    def __init__(self, als):
        """"""
        self._als = als
        self._A = als.A._M
        self._b = als.b._v
        self._freeze()

    def __call__(self, update_x=True):
        """direct solver."""
        x = spspalinalg.spsolve(self._A, self._b)
        if update_x:
            self._als._static.x.update(x)
        else:
            pass
        return x

    def gmres(self):
        """"""
