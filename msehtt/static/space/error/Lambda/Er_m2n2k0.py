# -*- coding: utf-8 -*-
r"""
"""
from phyem.src.config import RANK, MASTER_RANK, COMM
from phyem.tools.quadrature import quadrature

import numpy as np


def error__m2n2k0(tpm, cf, cochain, degree, error_type):
    """"""
    elements = tpm.composition
    error = list()
    for e in elements:
        element = elements[e]
        etype = element.etype
        if etype in (
            5,
            9,
            "orthogonal rectangle",
            'unique curvilinear quad',
            "unique msepy curvilinear quadrilateral",
            "unique msepy curvilinear triangle",
        ):
            element_error = _er220_msepy_quadrilateral_(
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


from phyem.msehtt.static.space.reconstruct.Lambda.Rc_m2n2k0 import ___rc220_msepy_quadrilateral___


def _er220_msepy_quadrilateral_(element, cf, local_cochain, degree, error_type):
    """"""
    p = element.degree_parser(degree)[0]
    p = (p[0] + 2, p[1] + 2)
    nodes, weights = quadrature(p, 'Gauss').quad
    x, y, u = ___rc220_msepy_quadrilateral___(element, degree, local_cochain, *nodes, ravel=False)
    exact_u = cf(x, y)[0]
    meshgrid = np.meshgrid(*nodes, indexing='ij')
    J = element.ct.Jacobian(meshgrid[0], meshgrid[1])
    if error_type == 'L2':

        diff = (u - exact_u) ** 2
        error = np.einsum(
            'ij, i, j -> ',
            J * diff, *weights,
            optimize='optimal'
        )
        return error

    else:
        raise NotImplementedError(error_type)
