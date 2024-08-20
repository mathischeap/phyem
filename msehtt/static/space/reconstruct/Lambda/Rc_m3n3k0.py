# -*- coding: utf-8 -*-
"""
"""
import numpy as np


def reconstruct_Lambda__m3n3k0(tpm, degree, cochain, xi, et, sg, ravel=False):
    """"""
    assert isinstance(xi, np.ndarray) and xi.ndim == 1, f"xi must be 1d array."
    assert isinstance(et, np.ndarray) and et.ndim == 1, f"eta must be 1d array."
    assert isinstance(sg, np.ndarray) and sg.ndim == 1, f"sg must be 1d array."
    assert np.min(xi) >= -1 and np.max(xi) <= 1, f"xi must be in [-1, 1]"
    assert np.min(et) >= -1 and np.max(et) <= 1, f"eta must be in [-1, 1]"
    assert np.min(sg) >= -1 and np.max(sg) <= 1, f"sg must be in [-1, 1]"
    elements = tpm.composition
    x, y, z, u = {}, {}, {}, {}
    for e in elements:
        element = elements[e]
        etype = element.etype
        local_cochain = cochain[e]
        if etype in ("orthogonal hexahedron", ):
            x[e], y[e], z[e], u[e] = ___rc330_msepy_quadrilateral___(
                element, degree, local_cochain, xi, et, sg, ravel=ravel
            )
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return (x, y, z), (u, )


def ___rc330_msepy_quadrilateral___(element, degree, local_cochain, xi, et, sg, ravel=False):
    """"""
    shape: list = [len(xi), len(et), len(sg)]
    xi_et_sg, bfs = element.bf('m3n3k0', degree, xi, et, sg)
    xyz = element.ct.mapping(*xi_et_sg)
    x, y, z = xyz
    v = np.einsum('ij, i -> j', bfs[0], local_cochain, optimize='optimal')

    if ravel:
        pass
    else:
        x = x.reshape(shape, order='F')
        y = y.reshape(shape, order='F')
        z = z.reshape(shape, order='F')
        v = v.reshape(shape, order='F')
    return x, y, z, v
