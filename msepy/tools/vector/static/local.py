# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
"""
import numpy as np
from tools.frozen import Frozen


class MsePyStaticLocalVector(Frozen):
    """"""
    def __init__(self, _2d_data, gathering_matrix):
        """"""
        self._gm = gathering_matrix
        self.data = _2d_data  # 2d numpy array or csr-sparse-matrix or None.
        # when _data is None, we can use this static vector to receive a vector later on (solution of Ax=b).
        self._freeze()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, _data):
        """"""
        if _data is None:
            self._data = _data
        elif isinstance(_data, np.ndarray):
            assert _data.shape == self._gm.shape
            self._data = _data
        else:
            raise Exception(f"msepy static local vector only accept 2d array or None.")

    def __getitem__(self, i):
        """When `self._data` is None, raise Error."""
        return self.data[i]

    @staticmethod
    def is_static():
        """static"""
        return True
