# -*- coding: utf-8 -*-
r"""Here we define a function that generate a random scipy sparse system Ax = b

# run this demo with, for example,
$ mpiexec -n 4 python _MPI/generic/py/linear_system/globalize/solvers/_petsc/test.py

"""
import sys
sys.path.append('./')

import time
import numpy as np
import scipy.sparse.linalg as sp_sparse_linalg
from scipy.sparse import random as sp_sparse_random
from numpy.random import rand as np_rand
from mpi4py import MPI
# ---------------------------------apisolve import-------------------------------
# import sys
# # sys.path.insert(0, '~/home/yizhang/apisolve_v1')
# # sys.path.insert(0, '/public/home/jzc/Pengyu/Basic_py/AAAapi')
# from scipy.sparse import csr_matrix, diags
# # from get_poisson import gp
from phmpi.generic.py.linear_system.globalize.solvers._petsc.api_solver_vff import ApiSolve

COMM = MPI.COMM_WORLD
RANK: int = COMM.Get_rank()
SIZE: int = COMM.Get_size()   # total amount fo ranks


def rank_random_scipy_sparse_Axb(shape, density=0.001):
    """Make a random Ax=b system."""
    assert isinstance(
        shape, int) and shape > 1, f"shape={shape} is wrong. It must be an integer greater than 1."
    assert 0 < density <= 1, f"density={density} is wrong. It must be in (0,1]."
    # first we make a random sparse matrix.
    A = sp_sparse_random(shape, shape, density=density, format='lil')
    A[range(shape), range(shape)] = 10 / SIZE
    # make the diagonal entries to be 10 (when sum up A) to make sure A_total is not singular and diagonal dominating.
    A = A.tocsc()
    b = np_rand(shape)
    return A, b


shape = 5000
density = 0.01
# A, b are distributed into ranks
A, b = rank_random_scipy_sparse_Axb(shape, density=density)

# we collect A and sum them up, A_total = sum(A), in rank 0
A_total = COMM.reduce(A, op=MPI.SUM, root=0)
# similarly, we collect b and sum them up in rank 0
b_total = COMM.reduce(b, op=MPI.SUM, root=0)

if RANK == 0:
    time0 = time.time()
    x0_py_direct = sp_sparse_linalg.gmres(A_total, b_total)   # scipy solver scheme
    time1 = time.time()


# def mpi_api_solver(A, b):
#     """"""
#     # TODO: to be implemented


# x0 = mpi_api_solver(A, b)
# # note that this function is called under the mpiexec environment, it takes A and b instead of A_total, b_total
# # x0 should be equal/close to `x0_py_direct`.


# apisolve
apisolve = ApiSolve(A, b)
# if you need the info of ksp and pc, use "-ksp_view"
direct_options = (
    "-ksp_type lgmres  "
    # "-ksp_atol 1e-6 "
    # "-ksp_rtol 1e-6 "
    # "-pc_type lu  "
    # "-pc_factor_mat_solver_type superlu_dist  "
    # "-ksp_initial_guess_nonzero false"
)
gmres_options = (
    "-ksp_type gmres "
    # "-pc_type none "
    # "-ksp_atol 1e-6 "
    # "-ksp_rtol 1e-6 "
    # "-ksp_max_it 200000"
)

# start solving
time2 = time.time()
x_direct = apisolve.solver(options=direct_options)
time3 = time.time()
x_iter = apisolve.solver(options=gmres_options)
time4 = time.time()
apisolve.lib.PetscFinalize()

# print(x_iter[:5])

if RANK == 0:
    # show the results
    print('------------------------------python print-------------------begin')
    print('-----------check A and b in python------------- \n')
    # print('A: \n', A, '\n')
    # print('b: \n', b, '\n')
    print('A shape', A.shape)
    print('b shape', b.shape)
    print('--------------show the results----------------- \n')
    print('python_direct_result: \n', x0_py_direct[:5], '\n')
    print('petsc_direct_result: \n', x_direct[:5], '\n')
    print('petsc_iter_result: \n', x_iter[:5], '\n')
    print('------------show the execution time------------')
    print('python_solver_duration: \n', (time1 - time0) * 1000, 'ms \n')
    print('apisolve_solver1_duration: \n', (time3 - time2) * 1000, 'ms \n')
    print('apisolve_solver2_duration: \n', (time4 - time3) * 1000, 'ms \n')
    print('------------------------------python print----------------------end')
