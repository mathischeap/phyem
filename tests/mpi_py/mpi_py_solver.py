# -*- coding: utf-8 -*-
r"""
$ mpiexec -n 4 python tests/mpi_py/mpi_py_solver.py
"""
import sys

if './' not in sys.path:
    sys.path.append('./')

import numpy as np
from scipy import sparse as sp_spa
from src.config import RANK, MASTER_RANK, COMM
from _MPI.generic.py.matrix.globalize.static import MPI_PY_Globalize_Static_Matrix
from _MPI.generic.py.vector.globalize.static import MPI_PY_Globalize_Static_Vector
from _MPI.generic.py.linear_system.globalize.solve import MPI_PY_Solve


if RANK != MASTER_RANK:
    Ar = np.random.rand(9, 9)
    br = np.random.rand(9)
else:
    Ar = np.zeros((9, 9))
    br = np.zeros(9)

Ar0 = COMM.gather(Ar, root=MASTER_RANK)
br0 = COMM.gather(br, root=MASTER_RANK)
if RANK == MASTER_RANK:
    A = np.array([(0.2, 1, -1, 0, 0.01, 0, 0, 0, -0.01),
                  (0.01, 0.3, 0, 0, 0, 0, 0, 0, 0),
                  (-0.1, 0, 0.4, 0, 0.3, 0, 0, 0, 0),
                  (0, 0, 0, 0.3, 0.6, 2, 0, 0, 0),
                  (0, -0.2, 0, 0, 0.4, 0, 0, 0, 1.1),
                  (0, 0, 0, -0.2, 0.1, 0.5, 0, 0, 0),
                  (1.2, 0, 0, 0, 0, 0, 0.4, 0.02, 3.0),
                  (0, 0, 0, 0, 0, 0, 2.0, 0.5, 0),
                  (0, 0, 0, 0, 0, 0, 0, 0.1, 0.6)])
    b = np.ones(9)

    AR0 = sum(Ar0)
    BR0 = sum(br0)
    Ar = A - AR0
    br = b - BR0

Ar = sp_spa.csc_matrix(Ar)
A = MPI_PY_Globalize_Static_Matrix(Ar)
b = MPI_PY_Globalize_Static_Vector(br)

M = A._gather()
V = b._gather()
if RANK == MASTER_RANK:
    np.testing.assert_array_almost_equal(
        M.toarray(),
        np.array([
            (0.2, 1, -1, 0, 0.01, 0, 0, 0, -0.01),
            (0.01, 0.3, 0, 0, 0, 0, 0, 0, 0),
            (-0.1, 0, 0.4, 0, 0.3, 0, 0, 0, 0),
            (0, 0, 0, 0.3, 0.6, 2, 0, 0, 0),
            (0, -0.2, 0, 0, 0.4, 0, 0, 0, 1.1),
            (0, 0, 0, -0.2, 0.1, 0.5, 0, 0, 0),
            (1.2, 0, 0, 0, 0, 0, 0.4, 0.02, 3.0),
            (0, 0, 0, 0, 0, 0, 2.0, 0.5, 0),
            (0, 0, 0, 0, 0, 0, 0, 0.1, 0.6)]
        )
    )
    np.testing.assert_array_almost_equal(V, np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))

solve = MPI_PY_Solve(A, b)
solve.x0 = 0
solve.package = 'py_mpi'

solve.scheme = 'gmres'
x, message, info = solve()
np.testing.assert_array_almost_equal(
    x,
    np.array([-3.23891085, 3.44129703, 1.7765975, -2.7063454, -0.11510028,
              0.94048189, 0.36495389, 0.54018445, 1.57663592])
)

solve.scheme = 'lgmres'
x, message, info = solve()
np.testing.assert_array_almost_equal(
    x,
    np.array([-3.23891085, 3.44129703, 1.7765975, -2.7063454, -0.11510028,
              0.94048189, 0.36495389, 0.54018445, 1.57663592])
)

if RANK == MASTER_RANK:
    print('\n>>> mpi-py-solver-passed!\n', flush=True)
