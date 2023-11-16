# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
import ctypes


import os
api_petsc_dir = os.path.dirname(__file__)


# def api_solve(A, b, scheme_name, maxiter=5, x0=None):
class ApiSolve():
    """
    With this class, we send `A` and `b` to a C/C++ program at the background,
    and solve Ax=b in it using the scheme named `scheme_name`, for example, `scheme_name='gmres'`.

    parameters
    ----------
    A : type A <class 'scipy.sparse.csr.csr_matrix'> (shape, shape)
    b : type b <class 'numpy.ndarray'> (shape,)
    scheme_name : gmres, lgmres, cg, bicg ...
    lib_route : the route of the extern library. Here we use lib_route = './lib_name.so'.
    maxiter :
    x0 : The initial guess for iterative schemes.
    """

    def __init__(self, A, b, lib_route=None):
        # load extern c/c++ library
        if lib_route is None:
            lib_route = api_petsc_dir + '/solve_mpivff.so'
        else:
            pass

        self.lib = ctypes.CDLL(lib_route)
        self.lib.solver.argtypes = [
            # information of the linear system
            ctypes.POINTER(ctypes.c_double),  # x_data_ptr
            ctypes.POINTER(ctypes.c_double),  # A_data_ptr
            ctypes.POINTER(ctypes.c_int),  # A_indices_ptr
            ctypes.POINTER(ctypes.c_int),  # A_indptr_ptr
            ctypes.POINTER(ctypes.c_double),  # b_data_ptr
            ctypes.c_int,  # shape
            # ctypes.c_int,                     # nnz
            ctypes.c_char_p  # options
        ]

        # ctypes获取数组指针
        # 转换A的稀疏格式
        if A.__class__.__name__ != 'csr_matrix':
            A = A.tocsr()

        self.A_data_ptr = A.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.A_indices_ptr = A.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.A_indptr_ptr = A.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        self.b_data_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.shape = len(b)

        # self.none_zeros = A.nnz # number of non-zero members in A
        # self.RANK = MPI.COMM_WORLD.Get_rank()
        # self.SIZE = MPI.COMM_WORLD.Get_size()

    def solver(self, options: str = "-ksp_type gmres -pc_type none", x0=None):
        # 如果x0为none，设置一个零向量，并生成ctypes指针
        if x0 is None:
            x_data_ptr = np.zeros(self.shape).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        else:
            x_data_ptr = x0.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        # 将options转为utf-8编码的字符串指针
        options = ctypes.c_char_p(options.encode('utf-8'))
        # 调用solver
        self.lib.solver(
            x_data_ptr,
            self.A_data_ptr,
            self.A_indices_ptr,
            self.A_indptr_ptr,
            self.b_data_ptr,
            self.shape,
            options
        )

        # 将返回的指针转为numpy数组
        return np.ctypeslib.as_array(x_data_ptr, shape=(self.shape,))
