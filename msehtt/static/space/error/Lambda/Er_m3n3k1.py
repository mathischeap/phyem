# -*- coding: utf-8 -*-
r"""
"""
from src.config import RANK, MASTER_RANK, COMM
from tools.quadrature import quadrature
import numpy as np


def error__m3n3k1(tpm, cf, cochain, degree, error_type):
    """"""
    elements = tpm.composition
    error = list()
    for e in elements:
        element = elements[e]
        etype = element.etype
        if etype in ("orthogonal hexahedron", ):
            element_error = _er331_msepy_quadrilateral_(
                element, cf, cochain[e], degree, error_type)

            error.append(element_error)
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")

    if error_type == 'L2':
        error = sum(error)
        error = COMM.gather(error, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            error = sum(error)
            L2_error = error ** 0.5
        else:
            L2_error = 0
        return COMM.bcast(L2_error, root=MASTER_RANK)
    else:
        raise NotImplementedError(f"error_type = {error_type}.")


from msehtt.static.space.reconstruct.Lambda.Rc_m3n3k1 import ___rc331_orthogonal_hexahedron___


def _er331_msepy_quadrilateral_(element, cf, local_cochain, degree, error_type):
    """"""
    p = element.degree_parser(degree)[0]
    p = (p[0] + 2, p[1] + 2, p[2] + 2)
    nodes, weights = quadrature(p, 'Gauss').quad
    x, y, z, u, v, w = ___rc331_orthogonal_hexahedron___(element, degree, local_cochain, *nodes, ravel=False)
    exact_u, exact_v, exact_w = cf(x, y, z)
    meshgrid = np.meshgrid(*nodes, indexing='ij')
    J = element.ct.Jacobian(*meshgrid)
    if error_type == 'L2':

        diff = (u - exact_u) ** 2 + (v - exact_v) ** 2 + (w - exact_w) ** 2
        error = np.einsum(
            'ijk, i, j, k -> ',
            J * diff, *weights,
            optimize='optimal'
        )
        return error

    else:
        raise NotImplementedError(error_type)
