# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from tools.quadrature import Quadrature


def reduce_Lambda__m2n2k1_outer(target, t, tpm, degree):
    """Reduce target at time `t` to m2n2k1 outer space of degree ``degree`` on partial mesh ``tpm``."""

    elements = tpm.composition
    cochain = {}
    for e in elements:
        element = elements[e]
        etype = element.etype
        if etype in ("orthogonal rectangle", "unique msepy curvilinear quadrilateral"):
            cochain[e] = ___221o_msepy_quadrilateral___(element, target, t, degree)
        else:
            raise NotImplementedError()
    return cochain


def ___221o_msepy_quadrilateral___(element, target, t, degree):
    """"""
    xi, et, edge_size_d, quad_weights = _msepy_data_preparation('x', degree)
    x, y = element.ct.mapping(xi, et)
    J = element.ct.Jacobian_matrix(xi, et)
    u, v = target[t](x, y)
    J00, J01 = J[0]
    J10, J11 = J[1]
    if isinstance(J01, (int, float)) and J01 == 0:
        vdx = J00 * v
    else:
        vdx = J00 * v - J10 * u
    cochain_dx = (
        np.einsum(
            'ij, i, j -> j',
            vdx, quad_weights[0], edge_size_d * 0.5,
            optimize='optimal'
        )
    )

    xi, et, edge_size_d, quad_weights = _msepy_data_preparation('y', degree)
    x, y = element.ct.mapping(xi, et)
    J = element.ct.Jacobian_matrix(xi, et)
    u, v = target[t](x, y)
    J00, J01 = J[0]
    J10, J11 = J[1]
    if isinstance(J01, (int, float)) and J01 == 0:
        vdy = J11 * u
    else:
        vdy = -J01 * v + J11 * u
    cochain_dy = (
        np.einsum(
            'ij, i, j -> j',
            vdy, quad_weights[1], edge_size_d * 0.5,
            optimize='optimal'
        )
    )
    return np.concatenate([cochain_dy, cochain_dx])


_cache_rd_221_dp_ = {}


def _msepy_data_preparation(d_, degree):
    """"""
    if isinstance(degree, int):
        p = (degree, degree)
        btype = 'Lobatto'
    else:
        raise NotImplementedError()

    key = str(p) + btype

    if key in _cache_rd_221_dp_:
        return _cache_rd_221_dp_[key]

    nodes = [Quadrature(_, category=btype).quad[0] for _ in p]
    qp = [p[0] + 2, p[1] + 2]
    quad_nodes, quad_weights = Quadrature(qp, category='Gauss').quad
    p_x, p_y = qp
    edges_size = [nodes[i][1:] - nodes[i][:-1] for i in range(2)]
    cell_nodes = [(0.5 * (edges_size[i][np.newaxis, :]) * (quad_nodes[i][:, np.newaxis] + 1)
                   + nodes[i][:-1]).ravel('F') for i in range(2)]

    if d_ == 'x':
        quad_xi = np.tile(cell_nodes[0], p[1] + 1).reshape(
            (p_x + 1, p[0] * (p[1] + 1)), order='F')
        quad_eta = np.repeat(nodes[1][np.newaxis, :], p[0], axis=0).ravel('F')
        quad_eta = quad_eta[np.newaxis, :].repeat(p_x + 1, axis=0)
        ES = np.tile(edges_size[0], p[1] + 1)
        data = quad_xi, quad_eta, ES, quad_weights

    elif d_ == 'y':
        quad_xi = np.tile(nodes[0], p[1])[np.newaxis, :].repeat(p_y + 1, axis=0)
        quad_eta = np.repeat(cell_nodes[1].reshape(
            (p_y + 1, p[1]), order='F'), p[0] + 1, axis=1)
        ES = np.repeat(edges_size[1], p[0] + 1)
        data = quad_xi, quad_eta, ES, quad_weights

    else:
        raise Exception()

    _cache_rd_221_dp_[key] = data

    return data
