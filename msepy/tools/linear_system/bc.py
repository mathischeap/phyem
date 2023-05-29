# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""
from tools.frozen import Frozen


class MsePyDynamicLinearSystemBoundaryCondition(Frozen):
    """"""

    def __init__(self, dynamic_ls, abstract_bc):
        self._dls = dynamic_ls
        self._abstract = abstract_bc
        self._freeze()


    def _bc_text(self):
        """"""
        return self._abstract._bc_text()

    def __len__(self):
        """how may bc defined?"""
        return len(self._abstract)
