# -*- coding: utf-8 -*-
"""
"""
import numpy as np


def reconstruct_Lambda__m2n2k1_outer(tpm, degree, cochain, xi, et, ravel=False):
    """"""
    assert isinstance(xi, np.ndarray) and xi.ndim == 1, f"xi must be 1d array."
    assert isinstance(et, np.ndarray) and et.ndim == 1, f"eta must be 1d array."
    assert np.min(xi) >= -1 and np.max(xi) <= 1, f"xi must be in [-1, 1]"
    assert np.min(et) >= -1 and np.max(et) <= 1, f"eta must be in [-1, 1]"
    elements = tpm.composition
    x, y, u, v = {}, {}, {}, {}
    for e in elements:
        element = elements[e]
        etype = element.etype
        local_cochain = cochain[e]
        if etype in ("orthogonal rectangle", "unique msepy curvilinear quadrilateral"):
            x[e], y[e], u[e], v[e] = ___rc221o_msepy_quadrilateral___(
                element, degree, local_cochain, xi, et, ravel=ravel
            )
        else:
            raise NotImplementedError()
    return (x, y), (u, v)


from msehtt.static.space.basis_function.Lambda.bf_m2n2k1 import ___bf221o_outer_msepy_quadrilateral___


def ___rc221o_msepy_quadrilateral___(element, degree, local_cochain, xi, et, ravel=False):
    """"""
    if isinstance(degree, int):
        px = py = degree
    else:
        raise NotImplementedError()
    shape: list = [len(xi), len(et)]
    xi_et, bfs = ___bf221o_outer_msepy_quadrilateral___(degree, xi, et)
    num_components = ((px+1) * py, px * (py+1))
    local_0 = local_cochain[:num_components[0]]
    local_1 = local_cochain[num_components[0]:]

    xy = element.ct.mapping(*xi_et)
    x, y = xy

    u = np.einsum('ij, i -> j', bfs[0], local_0, optimize='optimal')
    v = np.einsum('ij, i -> j', bfs[1], local_1, optimize='optimal')

    iJ = element.ct.inverse_Jacobian_matrix(*xi_et)
    iJ0, iJ1 = iJ
    iJ00, iJ01 = iJ0
    iJ10, iJ11 = iJ1

    if isinstance(iJ01, (int, float)) and iJ01 == 0:
        v0 = + u * iJ11
    else:
        v0 = + u * iJ11 - v * iJ01

    if isinstance(iJ10, (int, float)) and iJ10 == 0:
        v1 = + v * iJ00
    else:
        v1 = - u * iJ10 + v * iJ00

    if ravel:
        pass
    else:
        x = x.reshape(shape, order='F')
        y = y.reshape(shape, order='F')
        v0 = v0.reshape(shape, order='F')
        v1 = v1.reshape(shape, order='F')

    return x, y, v0, v1
