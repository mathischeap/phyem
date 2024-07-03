# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import linalg as spspalinalg
from time import time
from tools.miscellaneous.timer import MyTimer
from src.config import RANK, MASTER_RANK, COMM, MPI


def spsolve(A, b):  # I receive shells of A and b in order to have the freedom to clean the original data.
    r"""direct solver."""
    t_start = time()
    # --- x ------------------------------
    M = A.gather(root=MASTER_RANK)
    V = b.gather(root=MASTER_RANK)
    if RANK == MASTER_RANK:
        x = spspalinalg.spsolve(M, V)
    else:
        x = np.zeros(b.shape, dtype=float)
    COMM.Bcast([x, MPI.FLOAT], root=MASTER_RANK)
    # ====================================
    t_cost = time() - t_start
    t_cost = MyTimer.seconds2dhms(t_cost)
    message = f"Linear system of shape: {A.shape} <direct solver costs: {t_cost}> "
    info = {
        'total cost': t_cost,
    }
    message = COMM.bcast(message, root=MASTER_RANK)
    info = COMM.bcast(info, root=MASTER_RANK)
    return x, message, info
