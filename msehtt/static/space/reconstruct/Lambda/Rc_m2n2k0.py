# -*- coding: utf-8 -*-
"""
"""
import numpy as np


def reconstruct_Lambda__m2n2k0(tpm, degree, cochain, xi, et, ravel=False, element_range=None):
    """"""
    assert isinstance(xi, np.ndarray) and xi.ndim == 1, f"xi must be 1d array."
    assert isinstance(et, np.ndarray) and et.ndim == 1, f"eta must be 1d array."
    assert np.min(xi) >= -1 and np.max(xi) <= 1, f"xi must be in [-1, 1]"
    assert np.min(et) >= -1 and np.max(et) <= 1, f"eta must be in [-1, 1]"
    elements = tpm.composition
    x, y, u = {}, {}, {}

    if element_range is None:
        ELEMENTS_RANGE = elements
    else:
        ELEMENTS_RANGE = element_range

    for e in ELEMENTS_RANGE:
        element = elements[e]
        etype = element.etype
        local_cochain = cochain[e]
        if etype in (
                9,
                5,
                "orthogonal rectangle",
                "unique msepy curvilinear quadrilateral",
                "unique msepy curvilinear triangle",
                "unique curvilinear quad",
        ):
            x[e], y[e], u[e] = ___rc220_msepy_quadrilateral___(
                element, degree, local_cochain, xi, et, ravel=ravel
            )
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return (x, y), (u, )


def ___rc220_msepy_quadrilateral___(element, degree, local_cochain, xi, et, ravel=False):
    """"""
    shape: list = [len(xi), len(et)]
    xi_et, bfs = element.bf('m2n2k0', degree, xi, et)
    xy = element.ct.mapping(*xi_et)
    x, y = xy
    v = np.einsum('ij, i -> j', bfs[0], local_cochain, optimize='optimal')

    if ravel:
        pass
    else:
        x = x.reshape(shape, order='F')
        y = y.reshape(shape, order='F')
        v = v.reshape(shape, order='F')
    return x, y, v
