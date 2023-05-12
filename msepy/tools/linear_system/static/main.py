# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""

import sys

if './' not in sys.path:
    sys.path.append('/')
from tools.frozen import Frozen


class MsePyStaticLinearSystem(Frozen):
    """"""

    def __init__(self, A, x, b):
        self._A = A
        self._x = x
        self._b = b
        self._shape = None
        self._parse_msepy_static_ls_gathering_matrices()
        self._freeze()

    @property
    def shape(self):
        if self._shape is None:
            row_shape = len(self._A)
            col_shape = len(self._A[0])
            assert len(self._x) == col_shape and len(self._b) == row_shape, "A, x, b shape dis-match."
            self._shape = (row_shape, col_shape)
        return self._shape

    def _parse_msepy_static_ls_gathering_matrices(self):
        """"""
        row_shape, col_shape = self.shape
        row_gms = [None for _ in range(row_shape)]
        col_gms = [None for _ in range(col_shape)]

        for i in range(row_shape):
            for j in range(col_shape):
                A_ij = self._A[i][j]
                if A_ij is None:
                    pass
                else:
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

        assert None not in row_gms and None not in col_gms, f"miss some gathering matrices."
        self._row_gms = row_gms
        self._col_gms = col_gms

        # now we check gathering matrices in x.
        for j in range(col_shape):
            x_j = self._x[j]
            gm_j = x_j._gm
            assert gm_j is self._col_gms[j], f"by construction, this must be the case as we only construct" \
                                             f"gathering matrix once and store only once copy somewhere!"

        # now we check gathering matrices in b.
        for i in range(row_shape):
            b_i = self._b[i]
            if b_i is None:
                pass
            else:
                gm_i = b_i._gm
                assert gm_i is self._row_gms[i], f"by construction, this must be the case as we only construct" \
                                                 f"gathering matrix once and store only once copy somewhere!"

    @property
    def gathering_matrices(self):
        """return all gathering matrices; both row and col."""
        return self._row_gms, self._col_gms


if __name__ == '__main__':
    # python 
    pass
