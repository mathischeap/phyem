# -*- coding: utf-8 -*-
"""
"""
from src.config import RANK, MASTER_RANK, COMM
from tools.quadrature import quadrature
import numpy as np


def error__m2n2k1_outer(tpm, cf, cochain, degree, error_type):
    """"""
    return _compute_error__m2n2k1_(tpm, cf, cochain, degree, error_type, orientation='outer')


def error__m2n2k1_inner(tpm, cf, cochain, degree, error_type):
    """"""
    return _compute_error__m2n2k1_(tpm, cf, cochain, degree, error_type, orientation='inner')


def _compute_error__m2n2k1_(tpm, cf, cochain, degree, error_type, orientation='outer'):
    """"""
    elements = tpm.composition
    error = list()
    for e in elements:
        element = elements[e]
        etype = element.etype
        if etype in ("orthogonal rectangle", "unique msepy curvilinear quadrilateral"):
            if orientation == 'outer':
                element_error = _er221_msepy_quadrilateral_(
                    element, cf, cochain[e], degree, error_type, 'outer')
            elif orientation == 'inner':
                element_error = _er221_msepy_quadrilateral_(
                    element, cf, cochain[e], degree, error_type, 'inner')
            else:
                raise Exception()
            error.append(element_error)
        else:
            raise NotImplementedError()

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


from msehtt.static.space.reconstruct.Lambda.Rc_m2n2k1 import ___rc221o_msepy_quadrilateral___
from msehtt.static.space.reconstruct.Lambda.Rc_m2n2k1 import ___rc221i_msepy_quadrilateral___


def _er221_msepy_quadrilateral_(element, cf, local_cochain, degree, error_type, orientation):
    """"""
    p = element.degree_parser(degree)[0]
    p = (p[0] + 2, p[1] + 2)
    nodes, weights = quadrature(p, 'Gauss').quad
    if orientation == 'outer':
        x, y, u, v = ___rc221o_msepy_quadrilateral___(element, degree, local_cochain, *nodes, ravel=False)
    elif orientation == 'inner':
        x, y, u, v = ___rc221i_msepy_quadrilateral___(element, degree, local_cochain, *nodes, ravel=False)
    else:
        raise NotImplementedError()
    exact_u, exact_v = cf(x, y)
    meshgrid = np.meshgrid(*nodes, indexing='ij')
    J = element.ct.Jacobian(meshgrid[0], meshgrid[1])
    if error_type == 'L2':

        diff = (u - exact_u) ** 2 + (v - exact_v) ** 2
        error = np.einsum(
            'ij, i, j -> ',
            J * diff, *weights,
            optimize='optimal'
        )
        return error

    else:
        raise NotImplementedError(error_type)
