# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
"""

from tools.frozen import Frozen


class MsePyLocalVector(Frozen):
    """"""
    def __init__(self, _2d_data, gathering_matrix):
        """"""
        self._data = _2d_data  # 2d numpy array or csr-sparse-matrix or None.
        self._gm = gathering_matrix
        self._freeze()

    def __getitem__(self, i):
        """When `self._data` is None, raise Error."""
        return self._data[i]
