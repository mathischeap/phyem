# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 1:58 PM on 5/15/2023
"""
from tools.frozen import Frozen
from scipy.sparse import csr_matrix, csc_matrix
from numpy import diff
from msepy.tools.matrix.static.assembled import MsePyStaticAssembledMatrix


class MsePyStaticLocalMatrixAssemble(Frozen):
    """"""

    def __init__(self, M):
        """"""
        self._M = M
        self._freeze()

    def __call__(self, _format='csr'):
        """"""
        M = self._M

        gm_row = M._gm0_row
        gm_col = M._gm1_col

        dep = int(gm_row.num_dofs)
        wid = int(gm_col.num_dofs)

        ROW = list()
        COL = list()
        DAT = list()

        if _format == 'csc':
            SPA_MATRIX = csc_matrix
        elif _format == 'csr':
            SPA_MATRIX = csr_matrix
        else:
            raise Exception

        A = SPA_MATRIX((dep, wid))  # initialize a sparse matrix

        for i in M:
            Mi = M[i]  # all adjustments and customizations take effect
            indices = Mi.indices
            indptr = Mi.indptr
            data = Mi.data
            nums = diff(indptr)

            if Mi.__class__.__name__ == 'csc_matrix':
                for j, num in enumerate(nums):
                    idx = indices[indptr[j]:indptr[j+1]]
                    ROW.extend(gm_row[i][idx])
                    COL.extend([gm_col[i][j], ]*num)
            elif Mi.__class__.__name__ == 'csr_matrix':
                for j, num in enumerate(nums):
                    idx = indices[indptr[j]:indptr[j+1]]
                    ROW.extend([gm_row[i][j], ]*num)
                    COL.extend(gm_col[i][idx])
            else:
                raise Exception("I can not handle %r." % Mi)

            DAT.extend(data)

            if len(DAT) > 1e7:  # every 10 million data, we make it into sparse matrix.
                _ = SPA_MATRIX((DAT, (ROW, COL)), shape=(dep, wid))  # we make it into sparse

                del ROW, COL, DAT
                A += _
                del _
                ROW = list()
                COL = list()
                DAT = list()

        _ = SPA_MATRIX((DAT, (ROW, COL)), shape=(dep, wid))  # we make it into sparse

        del ROW, COL, DAT
        A += _

        return MsePyStaticAssembledMatrix(A, gm_row, gm_col)
