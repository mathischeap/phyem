# -*- coding: utf-8 -*-
r"""
"""
from numpy import linalg as np_linalg
import matplotlib.pyplot as plt
from tools.frozen import Frozen
from scipy.sparse import isspmatrix_csc, isspmatrix_csr
from src.config import RANK, MASTER_RANK, COMM


class MPI_PY_Globalize_Static_Matrix(Frozen):
    """A distributed shell of scipy sparse csr/csc matrix."""

    def __init__(self, M):
        assert isspmatrix_csc(M) or isspmatrix_csr(M), f"global matrix must be csc or csr."
        self._M = M
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return f"<MPI-PY-Globalize-Static-Matrix of shape {self.shape}{super_repr}>"

    @property
    def shape(self):
        return self._M.shape

    @property
    def M(self):
        return self._M

    def _gather(self, root=MASTER_RANK):
        """Gather the matrices and sum them up in the root rank. Return None in the non-root ranks."""
        M = COMM.gather(self._M, root=root)
        if RANK == root:
            M = sum(M)
        else:
            M = None
        return M

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
        M = self._gather(root=MASTER_RANK)
        if RANK == MASTER_RANK:
            M = M.toarray()
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
        else:
            return None

    @property
    def condition_number(self):
        """The condition number of this static assembled matrix."""
        M = self._gather(root=MASTER_RANK)
        if RANK == MASTER_RANK:
            M = M.toarray()
            cn = np_linalg.cond(M)
        else:
            cn = None

        return COMM.bcast(cn, root=MASTER_RANK)

    @property
    def rank(self):
        """compute the rank of this static assembled matrix"""
        M = self._gather(root=MASTER_RANK)
        if RANK == MASTER_RANK:
            M = M.toarray()
            rank = np_linalg.matrix_rank(M)
        else:
            rank = None
        return COMM.bcast(rank, root=MASTER_RANK)

    @property
    def num_singularities(self):
        """The amount of singular modes in this static assembled matrix."""
        return self.shape[0] - self.rank
