# -*- coding: utf-8 -*-
r"""
mpiexec -n 4 python phyem/tests/msehtt/solvers.py
"""
import numpy as np
from scipy import sparse as sp_spa

from phyem.src.config import RANK, MASTER_RANK, COMM
from phyem.msehtt.tools.matrix.static.global_ import MseHttGlobalMatrix
from phyem.msehtt.tools.vector.static.global_distributed import MseHttGlobalVectorDistributed
from phyem.msehtt.tools.linear_system.static.global_.main import MseHttLinearSystem


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

br = sp_spa.csr_matrix(br).T

Ar = sp_spa.csc_matrix(Ar)
A = MseHttGlobalMatrix(Ar)
b = MseHttGlobalVectorDistributed(br)

M = A.gather()
V = b.gather()
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


ls = MseHttLinearSystem(A, b)
solve = ls.solve
# x0, message, info = ls.solve(x0=0)

x, message0, _ = solve('gmres', x0=0)
np.testing.assert_array_almost_equal(
    x,
    np.array([-3.23891085, 3.44129703, 1.7765975, -2.7063454, -0.11510028,
              0.94048189, 0.36495389, 0.54018445, 1.57663592])
)

x, message1, _ = solve('lgmres', x0=0)
np.testing.assert_array_almost_equal(
    x,
    np.array([-3.23891085, 3.44129703, 1.7765975, -2.7063454, -0.11510028,
              0.94048189, 0.36495389, 0.54018445, 1.57663592])
)

x, message3, _ = solve('spsolve', x0=None)
np.testing.assert_array_almost_equal(
    x,
    np.array([-3.23891085, 3.44129703, 1.7765975, -2.7063454, -0.11510028,
              0.94048189, 0.36495389, 0.54018445, 1.57663592])
)

x, message2, _ = solve('ppsp', x0=None, clean=True)
np.testing.assert_array_almost_equal(
    x,
    np.array([-3.23891085, 3.44129703, 1.7765975, -2.7063454, -0.11510028,
              0.94048189, 0.36495389, 0.54018445, 1.57663592])
)
