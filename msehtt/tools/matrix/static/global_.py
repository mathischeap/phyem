# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import matplotlib.pyplot as plt
from numpy import linalg as np_linalg
from src.config import COMM, RANK, MASTER_RANK, MPI
import numpy as np
from scipy.sparse import dia_matrix

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

    def spy(self, markerfacecolor='k', markeredgecolor='g', markersize=6):
        """spy the assembled matrix.

        Parameters
        ----------
        markerfacecolor
        markeredgecolor
        markersize

        Returns
        -------

        """
        M = self.gather(root=MASTER_RANK)
        if RANK != MASTER_RANK:
            return
        fig = plt.figure()
        plt.spy(
            M,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            markersize=markersize
        )
        plt.tick_params(axis='both', which='major', direction='out')
        plt.tick_params(which='both', top=True, right=True, labelbottom=True, labelright=True)
        plt.show()
        return fig

    @property
    def rank_nnz(self):
        r"""The nnz of the rank M. Note that the nnz of the total matrix is not equal to reduce(rank_nnz, op=MPI.SUM)
        because some entries are shared by multiple ranks.
        """
        return self._M.nnz

    @property
    def condition_number(self):
        r"""The condition number of this static assembled matrix."""
        M = self.gather(root=MASTER_RANK)
        if RANK == MASTER_RANK:
            cn = np_linalg.cond(M.toarray())
        else:
            cn = 0
        return COMM.bcast(cn, root=MASTER_RANK)

    @property
    def rank(self):
        r"""compute the rank of this static assembled matrix"""
        M = self.gather(root=MASTER_RANK)
        if RANK == MASTER_RANK:
            rank = np_linalg.matrix_rank(M.toarray())
        else:
            rank = 0
        return COMM.bcast(rank, root=MASTER_RANK)

    @property
    def num_singularities(self):
        r"""The amount of singular modes in this static assembled matrix."""
        return self.shape[0] - self.rank

    def diagonal(self, k=0):
        r"""Returns the kth diagonal of the matrix."""
        diag = self._M.diagonal(k=k)
        DIAG = np.zeros_like(diag, dtype=float)
        COMM.Allreduce(diag, DIAG, op=MPI.SUM)
        DIAG[DIAG < 1] = 1
        DIAG = np.reciprocal(DIAG)
        M = len(DIAG)
        return dia_matrix((DIAG, 0), shape=(M, M))

    def __rmatmul__(self, other):
        """other @ self"""
        if other.__class__ is dia_matrix:
            M = other @ self._M
            return self.__class__(M, self._gm_row, self._gm_col)
        else:
            raise Exception()
