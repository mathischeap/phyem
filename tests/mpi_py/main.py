# -*- coding: utf-8 -*-
r"""
$ mpiexec -n 4 python tests/mpi_py/main.py

"""
import sys

if './' not in sys.path:
    sys.path.append('./')


__all__ = [
    "_mpi_py_solver",
]


import tests.mpi_py.mpi_py_solver as _mpi_py_solver
