# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.quadrature import quadrature


def reduce_Lambda__m3n3k3(cf_t, tpm, degree):
    """Reduce target at time `t` to m3n3k3 space of degree ``degree`` on partial mesh ``tpm``."""

    elements = tpm.composition
    cochain = {}
    for e in elements:
        element = elements[e]
        etype = element.etype
        if etype in ("orthogonal hexahedron", ):
            cochain[e] = ___rd333_msepy_quadrilateral___(element, cf_t, degree)
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return cochain


def ___rd333_msepy_quadrilateral___(element, cf_t, degree):
    """"""
    xi, et, sg, volume, quad_weights = _preparation_m3n3k3(degree)
    x, y, z = element.ct.mapping(xi, et, sg)
    J = element.ct.Jacobian(xi, et, sg)
    u = cf_t(x, y, z)[0]

    if isinstance(J, (int, float)):
        cochain_local = (
            np.einsum(
                'ijkm, j, k, m, i -> i',
                J * u, quad_weights[0], quad_weights[1], quad_weights[2], volume,
                optimize='optimal',
            )
        )
    else:
        cochain_local = (
            np.einsum(
                'ijkm, ijkm, j, k, m, i -> i',
                u, J, quad_weights[0], quad_weights[1], quad_weights[2], volume,
                optimize='optimal',
            )
        )

    return cochain_local


_cache_rd333_dp_ = {}
from msehtt.static.mesh.great.elements.types.orthogonal_hexahedron import MseHttGreatMeshOrthogonalHexahedronElement


def _preparation_m3n3k3(degree):
    """"""
    p, btype = MseHttGreatMeshOrthogonalHexahedronElement.degree_parser(degree)

    key = str(p) + btype
    if key in _cache_rd333_dp_:
        return _cache_rd333_dp_[key]

    quad_degree = [_ + 1 for _ in p]
    quad_nodes, quad_weights = quadrature(tuple(quad_degree), 'Gauss').quad

    nodes = [quadrature(_, btype).quad[0] for _ in p]
    num_basis = p[0] * p[1] * p[2]

    xi = np.zeros((num_basis, quad_degree[0] + 1, quad_degree[1] + 1, quad_degree[2] + 1))
    et = np.zeros((num_basis, quad_degree[0] + 1, quad_degree[1] + 1, quad_degree[2] + 1))
    si = np.zeros((num_basis, quad_degree[0] + 1, quad_degree[1] + 1, quad_degree[2] + 1))
    volume = np.zeros(num_basis)

    for k in range(p[2]):
        for j in range(p[1]):
            for i in range(p[0]):
                m = i + j*p[0] + k*p[0]*p[1]
                xi[m, ...] = (quad_nodes[0][:, np.newaxis].repeat(
                    quad_degree[1] + 1, axis=1
                )[:, :, np.newaxis].repeat(quad_degree[2] + 1, axis=2) + 1) \
                    * (nodes[0][i+1]-nodes[0][i])/2 + nodes[0][i]

                et[m, ...] = (quad_nodes[1][np.newaxis, :].repeat(
                    quad_degree[0] + 1, axis=0
                )[:, :, np.newaxis].repeat(quad_degree[2] + 1, axis=2) + 1) \
                    * (nodes[1][j+1]-nodes[1][j])/2 + nodes[1][j]

                si[m, ...] = (quad_nodes[2][np.newaxis, :].repeat(
                    quad_degree[1] + 1, axis=0
                )[np.newaxis, :, :].repeat(quad_degree[0] + 1, axis=0) + 1) \
                    * (nodes[2][k+1]-nodes[2][k])/2 + nodes[2][k]

                volume[m] = (nodes[0][i+1]-nodes[0][i]) \
                    * (nodes[1][j+1]-nodes[1][j]) \
                    * (nodes[2][k+1]-nodes[2][k])

    data = xi, et, si, volume * 0.125, quad_weights

    _cache_rd333_dp_[key] = data
    return data
