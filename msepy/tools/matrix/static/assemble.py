# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from scipy.sparse import csr_matrix, csc_matrix
from numpy import diff
from msepy.tools.matrix.static.assembled import MsePyStaticAssembledMatrix
# import numpy as np

_msepy_assembled_StaticMatrix_cache = {}
# we can cache the assembled matrices in case that it is the same for many or even all time steps.


class MsePyStaticLocalMatrixAssemble(Frozen):
    """"""

    @property
    def ___assembled_class___(self):
        return MsePyStaticAssembledMatrix

    def __init__(self, M):
        """"""
        self._M = M
        self._freeze()

    def __call__(self, format='csc', cache=None):
        """

        Parameters
        ----------
        format
        cache :
            We can manually cache the assembled matrix by set ``cache`` to be a string. When next time
            it sees the same `cache` it will return the cached matrix from the cache, i.e.,
            ``_msepy_assembled_StaticMatrix_cache``.

        Returns
        -------

        """
        if cache is not None:
            assert isinstance(cache, str), f" ``cache`` must a string."
            if cache in _msepy_assembled_StaticMatrix_cache:
                return _msepy_assembled_StaticMatrix_cache[cache]
            else:
                pass

        else:
            pass

        gm_row = self._M._gm0_row
        gm_col = self._M._gm1_col

        dep = int(gm_row.num_dofs)
        wid = int(gm_col.num_dofs)

        ROW = list()
        COL = list()
        DAT = list()

        if format == 'csc':
            SPA_MATRIX = csc_matrix
        elif format == 'csr':
            SPA_MATRIX = csr_matrix
        else:
            raise Exception

        # A = SPA_MATRIX((dep, wid))  # initialize a sparse matrix

        for i in self._M:

            Mi = self._M[i]  # all adjustments and customizations take effect
            indices = Mi.indices
            indptr = Mi.indptr
            data = Mi.data
            nums = diff(indptr)
            row = []
            col = []

            if Mi.__class__.__name__ == 'csc_matrix':
                for j, num in enumerate(nums):
                    idx = indices[indptr[j]:indptr[j+1]]
                    row.extend(gm_row[i][idx])
                    col.extend([gm_col[i][j], ]*num)

            elif Mi.__class__.__name__ == 'csr_matrix':
                for j, num in enumerate(nums):
                    idx = indices[indptr[j]:indptr[j+1]]
                    row.extend([gm_row[i][j], ]*num)
                    col.extend(gm_col[i][idx])

            else:
                raise Exception("I can not handle %r." % Mi)

            ROW.extend(row)
            COL.extend(col)
            DAT.extend(data)

            # if len(DAT) > 1e7:  # every 10 million data, we make it into a sparse matrix.
            #     _ = SPA_MATRIX((DAT, (ROW, COL)), shape=(dep, wid))  # we make it into sparse
            #
            #     del ROW, COL, DAT
            #     A += _
            #     del _
            #     ROW = list()
            #     COL = list()
            #     DAT = list()

        # _ = SPA_MATRIX((DAT, (ROW, COL)), shape=(dep, wid))  # we make it into sparse
        # del ROW, COL, DAT
        # A += _

        A = SPA_MATRIX((DAT, (ROW, COL)), shape=(dep, wid))
        A = self.___assembled_class___(A, gm_row, gm_col)
        if isinstance(cache, str):
            _msepy_assembled_StaticMatrix_cache[cache] = A
        return A
