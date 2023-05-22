""""""
from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix
from msepy.tools.gathering_matrix import RegularGatheringMatrix
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

    for i in range(row_shape):
        for j in range(col_shape):
            A_ij = A_2d_list[i][j]

            if A_ij is None:
                pass
            else:
                assert A_ij.__class__ is MsePyStaticLocalMatrix, f"A[{i}][{j}] is {A_ij.__class__}, wrong!"
                row_gm_i = A_ij._gm0_row
                col_gm_j = A_ij._gm1_col

                if row_gms[i] is None:
                    row_gms[i] = row_gm_i
                else:
                    assert row_gms[i] is row_gm_i, f"by construction, this must be the case as we only construct" \
                                                   f"gathering matrix once and store only once copy somewhere!"

                if col_gms[j] is None:
                    col_gms[j] = col_gm_j
                else:
                    assert col_gms[j] is col_gm_j, f"by construction, this must be the case as we only construct" \
                                                   f"gathering matrix once and store only once copy somewhere!"

    chain_row_gm = RegularGatheringMatrix(row_gms)
    chain_col_gm = RegularGatheringMatrix(col_gms)

    # only adjustments take effect. Customization will be skipped.
    M = _MsePyStaticLocalMatrixBmat(A_2d_list, (row_shape, col_shape))

    return MsePyStaticLocalMatrix(M, chain_row_gm, chain_col_gm, M.cache_key)


class _MsePyStaticLocalMatrixBmat(Frozen):
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
                    data[r][c] = Arc._get_data_adjusted(i)

        return sp_bmat(
            data, format='csr'
        )

    def cache_key(self, i):
        """Do this in real time."""
        row_shape, col_shape = self._shape
        keys = list()
        for r in range(row_shape):
            for c in range(col_shape):
                Arc = self._A[r][c]

                if Arc is None:
                    pass
                else:
                    if i in Arc.adjust:
                        return 'unique'
                    else:
                        keys.append(
                            Arc._cache_key(i)
                        )

        if all([_ == 'constant' for _ in keys]):
            return 'constant'
        elif 'unique' in keys:
            return 'unique'
        else:
            return ''.join(keys)
