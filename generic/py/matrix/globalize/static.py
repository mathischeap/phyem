# -*- coding: utf-8 -*-
r"""
"""
from numpy import linalg as np_linalg
import matplotlib.pyplot as plt
from tools.frozen import Frozen
from scipy.sparse import isspmatrix_csc, isspmatrix_csr


class Globalize_Static_Matrix(Frozen):
    """"""

    def __init__(self, M, gm_row, gm_col):
        """"""
        assert isspmatrix_csc(M) or isspmatrix_csr(M), f"global matrix must be csc or csr."
        assert M.shape == (gm_row.num_dofs, gm_col.num_dofs), f"matrix shape dis-match the gathering matrices"
        self._M = M
        self._gm_row = gm_row
        self._gm_col = gm_col
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return f"<Globalize_Static_Matrix of shape {self.shape}{super_repr}>"

    @property
    def shape(self):
        return self._M.shape

    @property
    def M(self):
        return self._M

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
        M = self._M
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
    def condition_number(self):
        """The condition number of this static assembled matrix."""
        M = self._M.toarray()
        cn = np_linalg.cond(M)
        return cn

    @property
    def rank(self):
        """compute the rank of this static assembled matrix"""
        M = self._M.toarray()
        rank = np_linalg.matrix_rank(M)
        return rank

    @property
    def num_singularities(self):
        """The amount of singular modes in this static assembled matrix."""
        return self.shape[0] - self.rank
