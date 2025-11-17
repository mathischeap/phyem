# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.src.config import COMM
from phyem.msehtt.static.space.mass_matrix.Lambda.MM_m3n3k3 import mass_matrix_Lambda__m3n3k3


def norm_Lambda__m3n3k3(tpm, degree, cochain, norm_type='L2', component_wise=False):
    """"""
    if norm_type == "L2":
        M = mass_matrix_Lambda__m3n3k3(tpm, degree)[0]
        rank_elements = tpm.composition
        local_norm_square = 0
        for e in rank_elements:
            e_cochain = cochain[e]
            e_mm = M[e]
            local_norm_square += np.sum(e_cochain * (e_mm @ e_cochain))
        norm = (sum(COMM.allgather(local_norm_square)))**0.5

        if component_wise:
            return [norm, ]
        else:
            return norm

    else:
        raise NotImplementedError(f"norm_type={norm_type} is not implemented")
