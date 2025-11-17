# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import dia_matrix
import matplotlib.pyplot as plt
from numpy import linalg as np_linalg
from scipy.sparse import isspmatrix_csc, isspmatrix_csr

from phyem.tools.frozen import Frozen
from phyem.src.config import COMM, RANK, MASTER_RANK, MPI
# from phyem.scipy.sparse import csc_matrix, csr_matrix
from phyem.msehtt.tools.gathering_matrix import MseHttGatheringMatrix


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

    def clean(self):
        r"""Replace the data by None (or empty sparse matrix?)."""
        # if self._dtype == 'csc':
        #     # noinspection PyUnresolvedReferences
        #     self._M = csc_matrix(self.shape)
        # elif self._dtype == 'csr':
        #     # noinspection PyUnresolvedReferences
        #     self._M = csr_matrix(self.shape)
        # else:
        #     raise Exception()
        self._M = None

    def value_at(self, i, j):
        r"""Return the value of M[i][j] in all ranks. So we find the value at each rank, and then
        allreduce with SUM.
        """
        vij = self._M[i, j]
        return COMM.allreduce(vij, op=MPI.SUM)

    def nnz_of_row(self, i):
        r"""return the number of non-zero values in row #i of the total matrix. So we first sum
        the contributions of row #i from all ranks, then check the number of nnz in this summation.
        """
        Mi = self._M[i]
        sum_Mi = COMM.reduce(Mi, root=MASTER_RANK, op=MPI.SUM)
        if RANK == MASTER_RANK:
            nnz = sum_Mi.nnz
        else:
            nnz = 0
        return COMM.bcast(nnz, root=MASTER_RANK)

    def nnz_of_col(self, i):
        r"""return the number of non-zero values in col #i of the total matrix. So we first sum
        the contributions of col #i from all ranks, then check the number of nnz in this summation.
        """
        Mi = self._M[:, i]
        sum_Mi = COMM.reduce(Mi, root=MASTER_RANK, op=MPI.SUM)
        if RANK == MASTER_RANK:
            nnz = sum_Mi.nnz
        else:
            nnz = 0
        return COMM.bcast(nnz, root=MASTER_RANK)

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
            return None
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
            # noinspection PyUnresolvedReferences
            cn = np_linalg.cond(M.toarray())
        else:
            cn = 0
        return COMM.bcast(cn, root=MASTER_RANK)

    @property
    def rank(self):
        r"""compute the rank of this static assembled matrix"""
        M = self.gather(root=MASTER_RANK)
        if RANK == MASTER_RANK:
            # noinspection PyUnresolvedReferences
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

    def ___find_essential_rows___(self):
        """These rows only have a singler entry 1 at on the diagonal."""
        local_essential_rows = list()
        for i, Mi in enumerate(self._M):
            assert isspmatrix_csr(Mi), f"must be"
            if Mi.nnz == 1 and i == Mi.indices[0] and Mi.data[0] == 1.:
                local_essential_rows.append(i)
            else:
                pass

        local_essential_rows = COMM.gather(local_essential_rows, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            LER = list()
            for _ in local_essential_rows:
                LER.extend(_)
            del local_essential_rows

            SET_LER = set(LER)

            if len(SET_LER) == len(LER):
                essential_rows = LER
            else:
                essential_rows = list()
                for i in SET_LER:
                    if LER.count(i) == 1:
                        essential_rows.append(i)
            essential_rows.sort()

        else:
            essential_rows = None
        essential_rows = COMM.bcast(essential_rows, root=MASTER_RANK)
        return essential_rows

    def ___find_essential_dof_coefficients___(self, b):
        r"""Consider Ax=b (self is A), we could find the solution of dofs with essential BC. We pick up all
        these dofs such that we can use them for example to initialize a better first guess for iterative
        solvers.

        Parameters
        ----------
        b

        Returns
        -------

        """
        from phyem.msehtt.tools.vector.static.global_distributed import MseHttGlobalVectorDistributed
        assert b.__class__ is MseHttGlobalVectorDistributed

        essential_rows = self.___find_essential_rows___()

        b_values = b._V[essential_rows]
        B_values = COMM.gather(b_values, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            b_values = sum(B_values)
        else:
            pass

        COMM.Bcast(b_values, root=MASTER_RANK)

        return essential_rows, b_values
