# -*- coding: utf-8 -*-
r"""
"""
import numpy as np


# ----------- INNER -------------------------------------------------------------------------------

def reconstruct_Lambda__m2n2k1_inner(tpm, degree, cochain, xi, et, ravel=False, element_range=None):
    """"""
    assert isinstance(xi, np.ndarray) and xi.ndim == 1, f"xi must be 1d array."
    assert isinstance(et, np.ndarray) and et.ndim == 1, f"eta must be 1d array."
    assert np.min(xi) >= -1 and np.max(xi) <= 1, f"xi must be in [-1, 1]"
    assert np.min(et) >= -1 and np.max(et) <= 1, f"eta must be in [-1, 1]"
    elements = tpm.composition
    x, y, u, v = {}, {}, {}, {}

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
                "orthogonal rectangle",
                "unique msepy curvilinear quadrilateral",
                'unique curvilinear quad',
        ):
            x[e], y[e], u[e], v[e] = ___rc221i_msepy_quadrilateral___(
                element, degree, local_cochain, xi, et, ravel=ravel
            )
        elif etype in (
                5,
                "unique msepy curvilinear triangle",
        ):
            x[e], y[e], u[e], v[e] = ___rc221i_vtu_5___(
                element, degree, local_cochain, xi, et, ravel=ravel
            )
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return (x, y), (u, v)


def ___rc221i_msepy_quadrilateral___(element, degree, local_cochain, xi, et, ravel=False):
    """"""
    p, _ = element.degree_parser(degree)
    px, py = p
    shape: list = [len(xi), len(et)]
    xi_et, bfs = element.bf('m2n2k1_inner', degree, xi, et)
    num_components = (px * (py+1), (px+1) * py)
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

    if isinstance(iJ10, (int, float)) and iJ10 == 0:
        v0 = + u * iJ00
    else:
        v0 = + u * iJ00 + v * iJ10

    if isinstance(iJ01, (int, float)) and iJ01 == 0:
        v1 = + v * iJ11
    else:
        v1 = + u * iJ01 + v * iJ11

    if ravel:
        pass
    else:
        x = x.reshape(shape, order='F')
        y = y.reshape(shape, order='F')
        v0 = v0.reshape(shape, order='F')
        v1 = v1.reshape(shape, order='F')

    return x, y, v0, v1


def ___rc221i_vtu_5___(element, degree, local_cochain, xi, et, ravel=False):
    """"""
    p, _ = element.degree_parser(degree)
    px, py = p
    shape: list = [len(xi), len(et)]
    xi_et, bfs = element.bf('m2n2k1_inner', degree, xi, et)
    num_components = (px * (py+1), px * py)
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

    if isinstance(iJ10, (int, float)) and iJ10 == 0:
        v0 = + u * iJ00
    else:
        v0 = + u * iJ00 + v * iJ10

    if isinstance(iJ01, (int, float)) and iJ01 == 0:
        v1 = + v * iJ11
    else:
        v1 = + u * iJ01 + v * iJ11

    if ravel:
        pass
    else:
        x = x.reshape(shape, order='F')
        y = y.reshape(shape, order='F')
        v0 = v0.reshape(shape, order='F')
        v1 = v1.reshape(shape, order='F')

    return x, y, v0, v1


# ----------- OUTER -------------------------------------------------------------------------------

def reconstruct_Lambda__m2n2k1_outer(tpm, degree, cochain, xi, et, ravel=False, element_range=None):
    """"""
    assert isinstance(xi, np.ndarray) and xi.ndim == 1, f"xi must be 1d array."
    assert isinstance(et, np.ndarray) and et.ndim == 1, f"eta must be 1d array."
    assert np.min(xi) >= -1 and np.max(xi) <= 1, f"xi must be in [-1, 1]"
    assert np.min(et) >= -1 and np.max(et) <= 1, f"eta must be in [-1, 1]"
    elements = tpm.composition
    x, y, u, v = {}, {}, {}, {}

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
                "orthogonal rectangle",
                "unique msepy curvilinear quadrilateral",
                'unique curvilinear quad',
        ):
            x[e], y[e], u[e], v[e] = ___rc221o_msepy_quadrilateral___(
                element, degree, local_cochain, xi, et, ravel=ravel
            )
        elif etype in (
                5,
                "unique msepy curvilinear triangle",
        ):
            x[e], y[e], u[e], v[e] = ___rc221o_vtu_5___(
                element, degree, local_cochain, xi, et, ravel=ravel
            )
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return (x, y), (u, v)


def ___rc221o_msepy_quadrilateral___(element, degree, local_cochain, xi, et, ravel=False):
    """"""
    p, _ = element.degree_parser(degree)
    px, py = p
    shape: list = [len(xi), len(et)]
    xi_et, bfs = element.bf('m2n2k1_outer', degree, xi, et)
    num_components = ((px+1) * py, px * (py+1))
    local_0 = local_cochain[:num_components[0]]
    local_1 = local_cochain[num_components[0]:]

    xy = element.ct.mapping(*xi_et)
    x, y = xy

    u = np.einsum('ij, i -> j', bfs[0], local_0, optimize='optimal')
    v = np.einsum('ij, i -> j', bfs[1], local_1, optimize='optimal')

    # ---------- IMPLEMENTATION 1 -----------------------------------------------------
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

    # ---------- IMPLEMENTATION 2 -----------------------------------------------------
    # iJ = element.ct.inverse_Jacobian(*xi_et)
    # JM = element.ct.Jacobian_matrix(*xi_et)
    #
    # j0, j1 = JM
    # j00, j01 = j0
    # j10, j11 = j1
    # v0 = iJ * (j00 * u + j01 * v)
    # v1 = iJ * (j10 * u + j11 * v)

    # ==================================================================================

    if ravel:
        pass
    else:
        x = x.reshape(shape, order='F')
        y = y.reshape(shape, order='F')
        v0 = v0.reshape(shape, order='F')
        v1 = v1.reshape(shape, order='F')

    return x, y, v0, v1


def ___rc221o_vtu_5___(element, degree, local_cochain, xi, et, ravel=False):
    """"""
    p, _ = element.degree_parser(degree)
    px, py = p
    shape: list = [len(xi), len(et)]
    xi_et, bfs = element.bf('m2n2k1_outer', degree, xi, et)
    num_components = (px * py, px * (py+1))
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
