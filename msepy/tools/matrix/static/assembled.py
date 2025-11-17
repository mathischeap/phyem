# -*- coding: utf-8 -*-
r"""
"""
from numpy import linalg as np_linalg
import matplotlib.pyplot as plt
from scipy.sparse import isspmatrix_csc, isspmatrix_csr

from phyem.tools.frozen import Frozen


class MsePyStaticAssembledMatrix(Frozen):
    """"""
    def __init__(self, M, gm_row, gm_col):
        """"""
        assert isspmatrix_csc(M) or isspmatrix_csr(M), f"M is {M.__class__} is not acceptable."
        self._M = M  # csc or csr matrix.
        self._gm_row = gm_row
        self._gm_col = gm_col
        self._shape = None
        self._freeze()

    @staticmethod
    def is_static():
        """static"""
        return True

    @property
    def shape(self):
        if self._shape is None:
            self._shape = self._M.shape
            assert self._shape == (self._gm_row.num_dofs, self._gm_col.num_dofs)
        return self._shape

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
