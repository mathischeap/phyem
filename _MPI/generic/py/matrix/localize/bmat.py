# -*- coding: utf-8 -*-
r"""
"""
from _MPI.generic.py.matrix.localize.static import MPI_PY_Localize_Static_Matrix
from _MPI.generic.py.gathering_matrix import MPI_PyGM
from tools.frozen import Frozen
from scipy.sparse import bmat as sp_bmat


def bmat(A_2d_list):
    """"""
    row_shape = len(A_2d_list)
    for Ai_ in A_2d_list:
        assert isinstance(Ai_, (list, tuple)), f"bmat must apply to 2d list or tuple."
    col_shape = len(A_2d_list[0])

    row_gms = [None for _ in range(row_shape)]
    col_gms = [None for _ in range(col_shape)]

    key_maps = list()
    for i in range(row_shape):
        for j in range(col_shape):
            A_ij = A_2d_list[i][j]

            if A_ij is None:
                pass
            else:
                assert A_ij.__class__ is MPI_PY_Localize_Static_Matrix, f"A[{i}][{j}] is {A_ij.__class__}, wrong!"

                key_maps.append(A_ij._key_map)

                row_gm_i = A_ij._gm_row
                col_gm_j = A_ij._gm_col

                if row_gms[i] is None:
                    row_gms[i] = row_gm_i
                else:
                    assert row_gms[i] == row_gm_i, f"by construction, this must be the case as we only construct" \
                                                   f"gathering matrix once and store only once copy somewhere!"

                if col_gms[j] is None:
                    col_gms[j] = col_gm_j
                else:
                    assert col_gms[j] == col_gm_j, f"by construction, this must be the case as we only construct" \
                                                   f"gathering matrix once and store only once copy somewhere!"

    chain_row_gm = MPI_PyGM(*row_gms)
    chain_col_gm = MPI_PyGM(*col_gms)

    # only adjustments take effect. Customization will be skipped.
    M = _MPI_PY_Bmat(A_2d_list, (row_shape, col_shape))

    return MPI_PY_Localize_Static_Matrix(M, chain_row_gm, chain_col_gm, raw_key_map=key_maps)


class _MPI_PY_Bmat(Frozen):
    """"""

    def __init__(self, A_2d_list, shape):
        """"""
        self._A = A_2d_list
        self._shape = shape
        self._freeze()

    def __call__(self, i):
        row_shape, col_shape = self._shape
        data = [[None for _ in range(col_shape)] for _ in range(row_shape)]
        for r in range(row_shape):
            for c in range(col_shape):
                Arc = self._A[r][c]
                if Arc is None:
                    pass
                else:
                    data[r][c] = Arc[i]  # ALL adjustments and customizations take effect!

        return sp_bmat(
            data, format='csr'
        )
