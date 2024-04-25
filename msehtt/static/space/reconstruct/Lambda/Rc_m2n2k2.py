# -*- coding: utf-8 -*-
"""
"""
import numpy as np


def reconstruct_Lambda__m2n2k2(tpm, degree, cochain, xi, et, ravel=False):
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
            x[e], y[e], u[e] = ___rc222_msepy_quadrilateral___(
                element, degree, local_cochain, xi, et, ravel=ravel
            )
        else:
            raise NotImplementedError()
    return (x, y), (u, )


from msehtt.static.space.basis_function.Lambda.bf_m2n2k2 import ___bf222_msepy_quadrilateral___ as bf222


def ___rc222_msepy_quadrilateral___(element, degree, local_cochain, xi, et, ravel=False):
    """"""
    shape: list = [len(xi), len(et)]
    xi_et, bfs = bf222(degree, xi, et)
    xy = element.ct.mapping(*xi_et)
    x, y = xy
    iJ = element.ct.inverse_Jacobian(*xi_et)
    if isinstance(iJ, (int, float)):
        v = (
            np.einsum(
                'i, ij -> j',
                iJ * local_cochain, bfs[0],
                optimize='optimal',
            )
        )
    else:
        v = (
            np.einsum(
                'i, j, ij -> j',
                local_cochain, iJ, bfs[0],
                optimize='optimal',
            )
        )

    if ravel:
        pass
    else:
        x = x.reshape(shape, order='F')
        y = y.reshape(shape, order='F')
        v = v.reshape(shape, order='F')

    return x, y, v
