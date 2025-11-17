# -*- coding: utf-8 -*-
r"""
"""
from numpy import ndarray

from phyem.tools.frozen import Frozen


class MsePyStaticAssembledVector(Frozen):
    """"""
    def __init__(self, v, gathering_matrix):
        """"""
        assert isinstance(v, ndarray) and v.ndim == 1, f"must be a 1-d array."
        self._v = v  # 1d numpy array.
        self._gm = gathering_matrix
        assert v.shape == (gathering_matrix.num_dofs, )
        self._freeze()

    @property
    def shape(self):
        """Shape of the vector, (x,)."""
        return self._v.shape

    @staticmethod
    def is_static():
        """static"""
        return True
