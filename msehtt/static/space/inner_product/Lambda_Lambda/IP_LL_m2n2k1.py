# -*- coding: utf-8 -*-
"""
"""
import numpy as np

from tools.quadrature import quadrature
from msehtt.static.space.reconstruct.Lambda.Rc_m2n2k1 import reconstruct_Lambda__m2n2k1_inner
from msehtt.static.space.reconstruct.Lambda.Rc_m2n2k1 import reconstruct_Lambda__m2n2k1_outer

from src.config import COMM, MPI


def ___Lambda_ip_Lambda_over_SameTPM_m2n2k1___(
        space1, degree1, cochain1,
        space2, degree2, cochain2,
        inner_type='L2'
):
    r""""""

    assert space1.tpm is space2.tpm, f"mesh must be the same."
    tpm = space1.tpm
    rank_elements = tpm.composition
    elements = tpm.composition

    orientation1 = space1.abstract.orientation
    orientation2 = space2.abstract.orientation

    if isinstance(degree1, int) and isinstance(degree2, int):
        quad_degree = max([degree1, degree2]) + 2
    else:
        raise NotImplementedError()

    quad = quadrature(quad_degree, category='Gauss')
    quad_nodes = quad.quad_nodes
    quad_weights = quad.quad_weights_ravel

    if orientation1 == 'inner':
        rc1 = reconstruct_Lambda__m2n2k1_inner(
            tpm, degree1, cochain1, quad_nodes, quad_nodes, ravel=False
        )
    elif orientation1 == 'outer':
        rc1 = reconstruct_Lambda__m2n2k1_outer(
            tpm, degree1, cochain1, quad_nodes, quad_nodes, ravel=False
        )
    else:
        raise Exception()

    if orientation2 == 'inner':
        rc2 = reconstruct_Lambda__m2n2k1_inner(
            tpm, degree2, cochain2, quad_nodes, quad_nodes, ravel=False
        )
    elif orientation2 == 'outer':
        rc2 = reconstruct_Lambda__m2n2k1_outer(
            tpm, degree2, cochain2, quad_nodes, quad_nodes, ravel=False
        )
    else:
        raise Exception()

    rc1 = rc1[1]
    rc2 = rc2[1]
    U1, V1 = rc1
    U2, V2 = rc2

    IPE = list()

    for e in rank_elements:
        u1 = U1[e]
        v1 = V1[e]
        u2 = U2[e]
        v2 = V2[e]

        element = elements[e]
        detJM = element.ct.Jacobian(quad_nodes, quad_nodes)

        if inner_type == 'L2':
            ipe = np.einsum(
                'ij,i,j->',
                (u1*u2 + v1*v2) * detJM, quad_weights, quad_weights,
                optimize='optimal'
            )
        else:
            raise NotImplementedError()
        IPE.append(ipe)

    IPE = sum(IPE)

    return COMM.allreduce(IPE, op=MPI.SUM)
