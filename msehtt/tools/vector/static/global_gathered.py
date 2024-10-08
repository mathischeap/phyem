# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import numpy as np
from msehtt.tools.gathering_matrix import MseHttGatheringMatrix


class MseHttGlobalVectorGathered(Frozen):
    """"""

    def __init__(self, V, gm=None):
        """"""
        assert isinstance(V, np.ndarray) and np.ndim(V) == 1, f"Gathered vector must be a 1d array."
        self._V = V
        if gm is None:
            pass
        else:
            assert gm.__class__ is MseHttGatheringMatrix, f"gathering matrix must be {MseHttGatheringMatrix}."
        self._dtype = 'vector-gathered'
        self._gm = gm
        self._freeze()

    @property
    def shape(self):
        """it must be 1d, so the shape is like (x,)."""
        return self._V.shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def V(self):
        """The gathered vector. A 1d array. Same in all ranks."""
        return self._V
