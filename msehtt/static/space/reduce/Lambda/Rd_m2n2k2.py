# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from tools.quadrature import Quadrature


def reduce_Lambda__m2n2k2(target, t, tpm, degree):
    """Reduce target at time `t` to m2n2k1 outer space of degree ``degree`` on partial mesh ``tpm``."""

    elements = tpm.composition
    cochain = {}
    for e in elements:
        element = elements[e]
        etype = element.etype
        if etype in ("orthogonal rectangle", "unique msepy curvilinear quadrilateral"):
            cochain[e] = ___rd222_msepy_quadrilateral___(element, target, t, degree)
        else:
            raise NotImplementedError()
    return cochain


def ___rd222_msepy_quadrilateral___(element, target, t, degree):
    """"""
    xi, et, volume, quad_weights = _preparation_m2n2k2(degree)
    x, y = element.ct.mapping(xi, et)
    J = element.ct.Jacobian(xi, et)
    u = target[t](x, y)[0]

    if isinstance(J, (int, float)):
        cochain_local = (
            np.einsum(
                'ijk, j, k, i -> i',
                J * u, quad_weights[0], quad_weights[1], volume,
                optimize='optimal',
            )
        )
    else:
        cochain_local = (
            np.einsum(
                'ijk, ijk, j, k, i -> i',
                u, J, quad_weights[0], quad_weights[1], volume,
                optimize='optimal',
            )
        )

    return cochain_local


_cache_rd222_dp_ = {}


def _preparation_m2n2k2(degree):
    """"""
    if isinstance(degree, int):
        p = (degree, degree)
        btype = 'Lobatto'
    else:
        raise NotImplementedError()

    key = str(p) + btype
    if key in _cache_rd222_dp_:
        return _cache_rd222_dp_[key]

    quad_degree = [p[0] + 2, p[1] + 2]
    nodes = [Quadrature(_, category=btype).quad[0] for _ in p]
    num_basis = p[0] * p[1]
    quad_nodes, quad_weights = Quadrature(quad_degree).quad
    magic_factor = 0.25
    xi = np.zeros((num_basis, quad_degree[0] + 1, quad_degree[1] + 1))
    et = np.zeros((num_basis, quad_degree[0] + 1, quad_degree[1] + 1))
    volume = np.zeros(num_basis)
    for j in range(p[1]):
        for i in range(p[0]):
            m = i + j*p[0]
            xi[m, ...] = (quad_nodes[0][:, np.newaxis].repeat(quad_degree[1] + 1, axis=1) + 1) \
                * (nodes[0][i+1]-nodes[0][i])/2 + nodes[0][i]
            et[m, ...] = (quad_nodes[1][np.newaxis, :].repeat(quad_degree[0] + 1, axis=0) + 1) \
                * (nodes[1][j+1]-nodes[1][j])/2 + nodes[1][j]
            volume[m] = (nodes[0][i+1]-nodes[0][i]) \
                * (nodes[1][j+1]-nodes[1][j]) * magic_factor
    data = xi, et, volume, quad_weights
    _cache_rd222_dp_[key] = data
    return data
