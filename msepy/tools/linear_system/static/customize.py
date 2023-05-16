# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 4:39 PM on 5/16/2023
"""
from tools.frozen import Frozen


class MsePyStaticLinearSystemCustomize(Frozen):
    """"""

    def __init__(self, ls):
        """"""
        self._ls = ls
        self._freeze()

    def set_dof(self, i, value):
        """Set the solution of dof #i to be `value`."""
        A = self._ls.A._mA
        b = self._ls.b._vb
        A.customize.identify_row(i)
        b.customize.set_value(i, value)
