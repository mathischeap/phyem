# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from src.config import RANK, MASTER_RANK, COMM
from tools.frozen import Frozen
from scipy.sparse import isspmatrix_csc
from msehtt.tools.gathering_matrix import MseHttGatheringMatrix


class MseHttGlobalVectorDistributed(Frozen):
    """The vector is distributed in all ranks. So the real vector is the summation of all vectors in
    all ranks.
    """

    def __init__(self, V, gm=None):
        """"""
        if isspmatrix_csc(V):
            assert V.shape[1] == 1, f"V must be of shape (x, 1)."
            V = V.toarray().ravel('F')
        elif isinstance(V, np.ndarray):
            if np.ndim(V) == 1:
                pass
            elif np.ndim(V) == 2 and V.shape[1] == 1:
                V = V[:, 0]
            else:
                raise Exception(f"V must be 1d array!")
        else:
            raise NotImplementedError()

        self._V = V
        if gm is None:
            pass
        else:
            assert gm.__class__ is MseHttGatheringMatrix, f"gathering matrix must be {MseHttGatheringMatrix}."
        self._gm = gm
        self._dtype = 'vector-distributed'
        self._freeze()

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} of shape {self.shape} @RANK{RANK}" + super_repr

    @property
    def shape(self):
        """the shape of the vector, must be like (x,)."""
        return self._V.shape

    @property
    def dtype(self):
        """The dtype indicator, must be 'distributed-vector'."""
        return self._dtype

    @property
    def V(self):
        """The local vector."""
        return self._V

    @property
    def gm(self):
        """The gathering matrix. It can be None. Then some functions cannot be fulfilled."""
        return self._gm

    def gather(self, root=MASTER_RANK):
        """Gather all V into one rank. Return None in other ranks."""
        all_V = COMM.gather(self._V, root=root)
        if RANK == root:
            return sum(all_V)
        else:
            return None
