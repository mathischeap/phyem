# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
"""

from tools.frozen import Frozen
from msepy.mesh.elements import _DataDictDistributor


class MsePyLocalMatrix(Frozen):
    """"""
    def __init__(self, data, gm_row, gm_col):
        """"""
        if data.__class__ is _DataDictDistributor:
            self._data = data  # csc or csr matrix.
        else:
            pass
        self._gmr = gm_row
        self._gmc = gm_col
        self._freeze()

    def __getitem__(self, i):
        """When `self._data` is None, raise Error."""
        return self._data[i]