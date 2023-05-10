# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
"""

from tools.frozen import Frozen


class MsePyStaticAssembledVector(Frozen):
    """"""
    def __init__(self, data, gathering_matrix):
        """"""
        self._data = data  # 1d numpy array or csc-sparse-matrix (shape (x,1)) or None.
        self._gm = gathering_matrix
        self._freeze()

    def __getitem__(self, i):
        """When `self._data` is `None`, raise Error."""
        return self._data[i]

    @staticmethod
    def is_static():
        """static"""
        return True
