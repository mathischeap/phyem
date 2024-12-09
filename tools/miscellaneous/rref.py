# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from sympy import Matrix


def rref(A, coe=1e-10, rzr=True):
    """


    Parameters
    ----------
    A
    coe :
        cut-off error. For element in the output whose abs value is lower than `coe`, we
        make it zero in the output.
    rzr :
        remove zero rows. Bool.
        If True, we remove rows of all zeros.

    Returns
    -------

    """
    if isinstance(A, np.ndarray):
        assert A.ndim == 2, f"rref needs a 2d array."
        rref = np.array(Matrix(A).rref()[0])
    else:
        raise NotImplementedError()

    rref[np.abs(rref) < coe] = 0

    if rzr:
        rref = rref[~np.all(rref == 0, axis=1)]
    else:
        pass

    return rref


if __name__ == '__main__':
    # python tools/miscellaneous/rref.py

    A = np.array([
        [1, 1, 1],
        [2, 2, 2],
        [7, 7, 7]
    ])

    print(rref(A))

    print(np.linalg.matrix_rank(A))
