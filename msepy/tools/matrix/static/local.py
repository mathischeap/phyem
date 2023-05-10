# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
"""
from scipy.sparse import issparse

from tools.frozen import Frozen
from msepy.mesh.elements import _DataDictDistributor


class MsePyStaticLocalMatrix(Frozen):
    """"""
    def __init__(self, data, gm_row, gm_col):
        """"""
        if data.__class__ is _DataDictDistributor:
            self._dtype = 'ddd'
            self._data = data  # csc or csr matrix.
        elif issparse(data):
            self._dtype = 'constant'
            self._data = data
        else:
            raise NotImplementedError(f"MsePyLocalMatrix cannot take data of type {data.__class__}.")
        self._gmr = gm_row
        self._gmc = gm_col
        self._freeze()

    def __getitem__(self, i):
        """When `self._data` is None, raise Error.

        Get the matrix for element #i.
        """
        if self._dtype == 'ddd':
            return self._data.get_data_of_element(i)
        elif self._dtype == 'constant':
            return self._data
        else:
            raise Exception()

    @staticmethod
    def is_static():
        """static"""
        return True
