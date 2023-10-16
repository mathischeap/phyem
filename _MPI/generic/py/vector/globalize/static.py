# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, COMM


class MPI_PY_Globalize_Static_Vector(Frozen):
    """A distributed shell of numpy 1d array."""

    def __init__(self, V):
        assert isinstance(V, np.ndarray) and np.ndim(V) == 1, f"V must be a 1-d ndarray."
        self._V = V
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return f"<MPI-PY-Globalize-Static-Vector of shape {self.shape}{super_repr}>"

    @property
    def shape(self):
        return self._V.shape

    @property
    def V(self):
        return self._V

    def _gather(self, root=MASTER_RANK):
        """"""
        V = COMM.gather(self.V, root=root)
        if RANK == root:
            V = sum(V)
        else:
            V = None
        return V
