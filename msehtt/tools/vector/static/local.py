# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from tools.frozen import Frozen
from msehtt.tools.gathering_matrix import MseHttGatheringMatrix


class EmptyDataError(Exception):
    """"""


class MsePyStaticLocalVector(Frozen):
    """"""
    def __init__(self, data, gathering_matrix):
        """"""
        assert gathering_matrix.__class__ is MseHttGatheringMatrix, f"I need a gathering matrix."
        self._gm = gathering_matrix
        self._receive_data(data)
        self._freeze()

    def _receive_data(self, data):
        """Do this such that data can be renewed.
        """
        # _2d_data: 2d numpy array or None.
        if data is None:
            self._dtype = 'None'
            self._data = None

        elif isinstance(data, (int, float)):
            self._dtype = "homogeneous"
            self._data = data

        elif isinstance(data, dict):
            self._dtype = "dict"
            for i in data:
                assert i in self._gm, f"element #{i} is not a local element"
                assert len(data[i]) == self._gm.num_local_dofs(i), f"num values in element #{i} is wrong."
            for i in self._gm:
                if self._gm.num_local_dofs(i) > 0:
                    assert i in data, f"data missing for element #{i}."
                else:
                    pass
            self._data = data

        elif callable(data):
            self._dtype = "realtime"
            self._data = data

        else:
            raise Exception(f"msepy static local vector data type wrong.")

    def _get_meta_data(self, i):
        """Get meta data for element #i"""
        assert i in self._gm, f"element #{i} is not a local element."
        if self._dtype == "None":
            raise EmptyDataError()
        elif self._dtype == 'homogeneous':
            return self._data * np.ones(self._gm.num_local_dofs(i))
        elif self._dtype == 'dict':
            if i in self._data:
                return self._data[i]
            else:
                return np.array([])
        elif self._dtype == "realtime":
            return self._data(i)

    @staticmethod
    def is_static():
        """static"""
        return True