# -*- coding: utf-8 -*-
"""
"""
import numpy as np


def reconstruct_Lambda__m3n3k2(tpm, degree, cochain, xi, et, sg, ravel=False):
    """"""
    assert isinstance(xi, np.ndarray) and xi.ndim == 1, f"xi must be 1d array."
    assert isinstance(et, np.ndarray) and et.ndim == 1, f"eta must be 1d array."
    assert isinstance(sg, np.ndarray) and sg.ndim == 1, f"sg must be 1d array."
    assert np.min(xi) >= -1 and np.max(xi) <= 1, f"xi must be in [-1, 1]"
    assert np.min(et) >= -1 and np.max(et) <= 1, f"eta must be in [-1, 1]"
    assert np.min(sg) >= -1 and np.max(sg) <= 1, f"sg must be in [-1, 1]"
    elements = tpm.composition
    x, y, z, u, v, w = {}, {}, {}, {}, {}, {}
    for e in elements:
        element = elements[e]
        etype = element.etype
        local_cochain = cochain[e]
        if etype in ("orthogonal hexahedron", ):
            x[e], y[e], z[e], u[e], v[e], w[e] = ___rc332_msepy_quadrilateral___(
                element, degree, local_cochain, xi, et, sg, ravel=ravel
            )
        else:
            raise NotImplementedError()
    return (x, y, z), (u, v, w)


def ___rc332_msepy_quadrilateral___(element, degree, local_cochain, xi, et, sg, ravel=False):
    """"""
    p, _ = element.degree_parser(degree)
    px, py, pz = p
    shape: list = [len(xi), len(et), len(sg)]
    xi_et_sg, bfs = element.bf('m3n3k2', degree, xi, et, sg)
    num_components = [(px+1) * py * pz, px * (py+1) * pz, px * py * (pz+1)]
    local_0 = local_cochain[:num_components[0]]
    local_1 = local_cochain[num_components[0]:(num_components[0]+num_components[1])]
    local_2 = local_cochain[-num_components[2]:]

    xyz = element.ct.mapping(*xi_et_sg)
    x, y, z = xyz

    u = np.einsum('ij, i -> j', bfs[0], local_0, optimize='optimal')
    v = np.einsum('ij, i -> j', bfs[1], local_1, optimize='optimal')
    w = np.einsum('ij, i -> j', bfs[2], local_2, optimize='optimal')

    ij = element.ct.inverse_Jacobian_matrix(*xi_et_sg)

    if element.etype == "orthogonal hexahedron":
        v0 = u*ij[1][1]*ij[2][2]
        v1 = v*ij[2][2]*ij[0][0]
        v2 = w*ij[0][0]*ij[1][1]
    else:
        v0 = \
            u*(ij[1][1]*ij[2][2] - ij[1][2]*ij[2][1]) + \
            v*(ij[2][1]*ij[0][2] - ij[2][2]*ij[0][1]) + \
            w*(ij[0][1]*ij[1][2] - ij[0][2]*ij[1][1])
        v1 = \
            u*(ij[1][2]*ij[2][0] - ij[1][0]*ij[2][2]) + \
            v*(ij[2][2]*ij[0][0] - ij[2][0]*ij[0][2]) + \
            w*(ij[0][2]*ij[1][0] - ij[0][0]*ij[1][2])
        v2 = \
            u*(ij[1][0]*ij[2][1] - ij[1][1]*ij[2][0]) + \
            v*(ij[2][0]*ij[0][1] - ij[2][1]*ij[0][0]) + \
            w*(ij[0][0]*ij[1][1] - ij[0][1]*ij[1][0])

    if ravel:
        pass
    else:
        x = x.reshape(shape, order='F')
        y = y.reshape(shape, order='F')
        z = z.reshape(shape, order='F')
        v0 = v0.reshape(shape, order='F')
        v1 = v1.reshape(shape, order='F')
        v2 = v2.reshape(shape, order='F')

    return x, y, z, v0, v1, v2
