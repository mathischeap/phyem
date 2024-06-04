# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from src.config import COMM
from msehtt.static.space.mass_matrix.Lambda.MM_m3n3k2 import mass_matrix_Lambda__m3n3k2


def norm_Lambda__m3n3k2(tpm, degree, cochain, norm_type='L2'):
    """"""
    if norm_type == "L2":
        M = mass_matrix_Lambda__m3n3k2(tpm, degree)[0]
        rank_elements = tpm.composition
        local_norm_square = 0
        for e in rank_elements:
            e_cochain = cochain[e]
            e_mm = M[e]
            local_norm_square += np.sum(e_cochain * (e_mm @ e_cochain))
        return (sum(COMM.allgather(local_norm_square)))**0.5

    else:
        raise NotImplementedError(f"norm_type={norm_type} is not implemented")
