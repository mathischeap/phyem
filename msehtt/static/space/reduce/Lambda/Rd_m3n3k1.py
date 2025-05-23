# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.quadrature import quadrature


# ----------------- INNER ---------------------------------------------------------------------------------

def reduce_Lambda__m3n3k1(cf_t, tpm, degree, element_range=None):
    """Reduce target at time `t` to m2n2k1 outer space of degree ``degree`` on partial mesh ``tpm``."""

    elements = tpm.composition
    cochain = {}

    if element_range is None:
        ELEMENT_RANGE = elements
    else:
        ELEMENT_RANGE = element_range

    for e in ELEMENT_RANGE:
        element = elements[e]
        etype = element.etype
        if etype in ("orthogonal hexahedron", ):
            cochain[e] = ___331_msepy_orthogonal___(element, cf_t, degree)
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    return cochain


def ___331_raw_msepy_orthogonal___(element, cf_t, degree):
    """"""
    xi, et, sg, edge_size_d, quad_weights = _msepy_data_preparation('x', degree)
    x, y, z = element.ct.mapping(xi, et, sg)
    J = element.ct.Jacobian_matrix(xi, et, sg)
    u, v, w = cf_t(x, y, z)
    J00 = J[0][0]
    J10 = J[1][0]
    J20 = J[2][0]

    vdx = J00 * u + J10 * v + J20 * w
    cochain_dx = (
        np.einsum(
            'ij, i, j -> j',
            vdx, quad_weights[0], edge_size_d,
            optimize='optimal'
        )
    )

    xi, et, sg, edge_size_d, quad_weights = _msepy_data_preparation('y', degree)
    x, y, z = element.ct.mapping(xi, et, sg)
    J = element.ct.Jacobian_matrix(xi, et, sg)
    u, v, w = cf_t(x, y, z)
    J01 = J[0][1]
    J11 = J[1][1]
    J21 = J[2][1]
    vdy = J01 * u + J11 * v + J21 * w

    cochain_dy = (
        np.einsum(
            'ij, i, j -> j',
            vdy, quad_weights[1], edge_size_d,
            optimize='optimal'
        )
    )

    xi, et, sg, edge_size_d, quad_weights = _msepy_data_preparation('z', degree)
    x, y, z = element.ct.mapping(xi, et, sg)
    J = element.ct.Jacobian_matrix(xi, et, sg)
    u, v, w = cf_t(x, y, z)
    J02 = J[0][2]
    J12 = J[1][2]
    J22 = J[2][2]
    vdz = J02 * u + J12 * v + J22 * w

    cochain_dz = (
        np.einsum(
            'ij, i, j -> j',
            vdz, quad_weights[2], edge_size_d,
            optimize='optimal'
        )
    )

    return cochain_dx, cochain_dy, cochain_dz


def ___331_msepy_orthogonal___(element, cf_t, degree):
    cochain_dx, cochain_dy, cochain_dz = ___331_raw_msepy_orthogonal___(element, cf_t, degree)
    if 'm3n3k1' in element.dof_reverse_info:
        face_indices = element.dof_reverse_info['m3n3k1']

        for fi in face_indices:
            component, local_dofs = element.find_local_dofs_on_face(
                indicator='m3n3k1', degree=degree, face_index=fi, component_wise=True
            )
            if component == 0:
                cochain_dx[local_dofs] = - cochain_dx[local_dofs]
            elif component == 1:
                cochain_dy[local_dofs] = - cochain_dy[local_dofs]
            elif component == 2:
                cochain_dz[local_dofs] = - cochain_dz[local_dofs]
            else:
                raise Exception()
    else:
        pass

    return np.concatenate([cochain_dx, cochain_dy, cochain_dz])


_cache331_data_ = {}
from msehtt.static.mesh.great.elements.types.orthogonal_hexahedron import MseHttGreatMeshOrthogonalHexahedronElement


def _msepy_data_preparation(d_, degree):
    """
    Parameters
    ----------
    d_ : str, optional
        'x', 'y' or 'z'.
    """

    p, btype = MseHttGreatMeshOrthogonalHexahedronElement.degree_parser(degree)
    key = d_ + str(p) + btype

    if key in _cache331_data_:
        data = _cache331_data_[key]
    else:
        quad_degree = [_ + 2 for _ in p]
        quad_nodes, quad_weights = quadrature(tuple(quad_degree), 'Gauss').quad
        quad_num_nodes = [len(quad_nodes_i) for quad_nodes_i in quad_nodes]
        nodes = [quadrature(_, btype).quad[0] for _ in p]

        sbn0 = nodes[0]
        sbn1 = nodes[1]
        sbn2 = nodes[2]

        if d_ == 'x':
            a = sbn0[1:] - sbn0[:-1]
            a = a.ravel('F')
            b = (p[1]+1)*(p[2]+1)
            edge_size_x = np.tile(a, b)
            snb_x = b * p[0]
            D = quad_nodes[0][:, np.newaxis].repeat(snb_x, axis=1) + 1
            assert np.shape(D)[1] == len(edge_size_x)
            xi1 = D * edge_size_x / 2
            xi2 = np.tile(sbn0[:-1], b)
            xi = xi1 + xi2
            eta = np.tile(np.tile(sbn1[:, np.newaxis].repeat(quad_num_nodes[0], axis=1).T,
                                  (p[0], 1)).reshape((quad_num_nodes[0], p[0]*(p[1]+1)), order='F'),
                          (1, p[2]+1))
            sigma = sbn2.repeat(p[0]*(p[1]+1))[np.newaxis, :].repeat(
                quad_num_nodes[0], axis=0)
            data = [xi, eta, sigma, edge_size_x * 0.5, quad_weights]

        elif d_ == 'y':
            edge_size_y = np.tile(np.repeat((sbn1[1:] - sbn1[:-1]),
                                            p[0]+1), p[2]+1)
            xi = np.tile(sbn0, p[1]*(p[2]+1))[np.newaxis, :].repeat(
                quad_num_nodes[1], axis=0)
            snb_y = (p[0] + 1) * p[1] * (p[2] + 1)
            eta1 = (quad_nodes[1][:, np.newaxis].repeat(snb_y, axis=1) + 1) * edge_size_y / 2
            eta2 = np.tile(np.repeat(sbn1[:-1], (p[0]+1)), (p[2]+1))
            eta = eta1 + eta2
            sigma = sbn2.repeat(p[1]*(p[0]+1))[np.newaxis, :].repeat(
                quad_num_nodes[1], axis=0)
            data = [xi, eta, sigma, edge_size_y * 0.5, quad_weights]

        elif d_ == 'z':
            edge_size_z = np.repeat((sbn2[1:] - sbn2[:-1]),
                                    p[0]+1).repeat(p[1]+1)
            xi = np.tile(sbn0, (p[1]+1)*(p[2]))[np.newaxis, :].repeat(
                quad_num_nodes[2], axis=0)
            eta = np.tile(np.repeat(sbn1, (p[0]+1)), p[2])[np.newaxis, :].repeat(
                quad_num_nodes[2], axis=0)
            snb_z = (p[0] + 1) * (p[1] + 1) * p[2]
            sigma1 = (quad_nodes[2][:, np.newaxis].repeat(snb_z, axis=1) + 1) * edge_size_z / 2
            sigma2 = sbn2[:-1].repeat((p[0]+1)*(p[1]+1))
            sigma = sigma1 + sigma2
            data = [xi, eta, sigma, edge_size_z * 0.5, quad_weights]

        else:
            raise Exception()

        _cache331_data_[key] = data

    return data
