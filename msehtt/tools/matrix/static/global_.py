# -*- coding: utf-8 -*-
r"""
"""
import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from src.config import COMM, RANK, MASTER_RANK

from scipy.sparse import isspmatrix_csc, isspmatrix_csr
from msehtt.tools.gathering_matrix import MseHttGatheringMatrix


class MseHttGlobalMatrix(Frozen):
    """"""

    def __init__(self, M, gm_row=None, gm_col=None):
        """The true M is the summation of all `M` across all ranks."""
        assert isspmatrix_csc(M) or isspmatrix_csr(M), f"M must be csr or csc."
        self._dtype = 'csc' if isspmatrix_csc(M) else 'csr'
        shape = M.shape
        all_shapes = COMM.gather(shape, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            assert all([_ == shape for _ in all_shapes]), f"M must be of same shape in all ranks."
        else:
            pass
        self._shape = shape
        self._M = M

        if gm_row is None:
            pass
        else:
            assert gm_row.__class__ is MseHttGatheringMatrix, f"gathering matrix must be {MseHttGatheringMatrix}."
        self._gm_row = gm_row
        if gm_col is None:
            pass
        else:
            assert gm_col.__class__ is MseHttGatheringMatrix, f"gathering matrix must be {MseHttGatheringMatrix}."
        self._gm_row, self._gm_col = gm_row, gm_col
        self._freeze()

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} of shape {self.shape} @RANK{RANK}" + super_repr

    @property
    def shape(self):
        """The shape of the global matrix."""
        return self._shape

    @property
    def dtype(self):
        """csc or csr."""
        return self._dtype

    @property
    def M(self):
        """The local matrix."""
        return self._M

    @property
    def gm_row(self):
        """The gathering matrix referring to the row (axis-0) of the matrix.

        It could be None, then some functions cannot be fulfilled.
        """
        return self._gm_row

    @property
    def gm_col(self):
        """The gathering matrix referring to the colum (axis-1) of the matrix.

        It could be None, then some functions cannot be fulfilled.
        """
        return self._gm_col

    def gather(self, root=MASTER_RANK):
        """Gather M into one rank."""
        all_M = COMM.gather(self._M, root=root)
        if RANK == root:
            return sum(all_M)
        else:
            return None
