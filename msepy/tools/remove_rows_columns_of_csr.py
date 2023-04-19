# -*- coding: utf-8 -*-

import sys
import numpy as np

if './' not in sys.path:
    sys.path.append('./')

from scipy.sparse import csr_matrix


def delete_from_csr(mat, _rows, _cols):
    """Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``)
    from the CSR sparse matrix ``mat``.

    WARNING: Indices of altered axes are reset in the returned matrix
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    if isinstance(_rows, int):
        rows = [_rows, ]
    else:
        rows = _rows

    if isinstance(_cols, int):
        cols = [_cols, ]
    else:
        cols = _cols

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:, col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:, mask]
    else:
        return mat


if __name__ == '__main__':
    # python msepy/tools/remove_rows_columns_of_csr.py

    a = csr_matrix((10, 10))
    b = delete_from_csr(a, [0,1,2], [3,4,5,6])
    print(b.shape)