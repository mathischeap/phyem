# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHttStaticLinearSystemCustomize(Frozen):
    """"""

    def __init__(self, sls):
        """"""
        self._sls = sls
        self._freeze()

    def set_dof(self, global_dof, value):
        """"""
        A = self._sls.A._mA
        b = self._sls.b._vb
        A.customize.identify_row(global_dof)
        b.customize.set_value(global_dof, value)
