# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from src.config import COMM, MPI
from tools.quadrature import quadrature
from msehtt.static.space.mass_matrix.Lambda.MM_m3n3k1 import mass_matrix_Lambda__m3n3k1


def norm_Lambda__m3n3k1(tpm, degree, cochain, norm_type='L2', component_wise=False):
    """"""
    if component_wise:
        return ___norm_Lambda__m3n3k1_component_wise___(tpm, degree, cochain, norm_type=norm_type)
    else:
        pass

    if norm_type == "L2":
        M = mass_matrix_Lambda__m3n3k1(tpm, degree)[0]
        rank_elements = tpm.composition
        local_norm_square = 0
        for e in rank_elements:
            e_cochain = cochain[e]
            e_mm = M[e]
            local_norm_square += np.sum(e_cochain * (e_mm @ e_cochain))
        return (sum(COMM.allgather(local_norm_square)))**0.5

    else:
        raise NotImplementedError(f"norm_type={norm_type} is not implemented")


from msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElement
from msehtt.static.space.reconstruct.Lambda.Rc_m3n3k1 import reconstruct_Lambda__m3n3k1


def ___norm_Lambda__m3n3k1_component_wise___(tpm, degree, cochain, norm_type='L2'):
    r""""""
    p, _ = MseHttGreatMeshBaseElement.degree_parser(degree, m=3, n=3)
    qp = tuple([_ + 2 for _ in p])
    quad = quadrature(qp, 'Gauss')
    nodes, weights = quad.quad
    elements = tpm.composition
    U, V, W = reconstruct_Lambda__m3n3k1(tpm, degree, cochain, *nodes, ravel=False)[1]
    local_norm_U = []
    local_norm_V = []
    local_norm_W = []
    if norm_type == 'L2':
        for i in U:
            element = elements[i]
            detJM = element.ct.Jacobian(*nodes)
            local_norm_U.append(
                np.einsum('ijk, i, j, k ->', U[i]**2 * detJM, *weights, optimize='optimal')
            )
            local_norm_V.append(
                np.einsum('ijk, i, j, k ->', V[i]**2 * detJM, *weights, optimize='optimal')
            )
            local_norm_W.append(
                np.einsum('ijk, i, j, k ->', W[i]**2 * detJM, *weights, optimize='optimal')
            )
    else:
        raise NotImplementedError(f"___norm_Lambda__m3n3k1_component_wise___ not implemented for norm_type={norm_type}")

    local_norm_U = sum(local_norm_U)
    local_norm_V = sum(local_norm_V)
    local_norm_W = sum(local_norm_W)

    global_norm_U = COMM.allreduce(local_norm_U, op=MPI.SUM)
    global_norm_V = COMM.allreduce(local_norm_V, op=MPI.SUM)
    global_norm_W = COMM.allreduce(local_norm_W, op=MPI.SUM)

    return global_norm_U, global_norm_V, global_norm_W
