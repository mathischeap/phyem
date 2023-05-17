# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 5:41 PM on 5/17/2023
"""
from scipy.sparse import linalg as spspalinalg
from tools.frozen import Frozen


class MsePyStaticLinearSystemAssembledSolve(Frozen):
    """"""
    def __init__(self, als):
        """"""
        self._als = als
        self._A = als.A._M
        self._b = als.b._v
        print(self._A.shape)
        self._freeze()

    def __call__(self, update_x=True):
        """direct solver."""
        x = spspalinalg.spsolve(self._A, self._b)
        if update_x:
            self._als._static.x.update(x)
        else:
            pass
        return x

    def gmres(self, update_x=True, **kwargs):
        """"""
        x, info = spspalinalg.gmres(
            self._A, self._b, **kwargs
        )
        if update_x:
            self._als._static.x.update(x)
        else:
            pass

        return x, info
