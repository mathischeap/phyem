# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.quadrature import quadrature


# ----------------- INNER ---------------------------------------------------------------------------------

def reduce_Lambda__m3n3k2(cf_t, tpm, degree):
    """Reduce target at time `t` to m3n3k2 space of degree ``degree`` on partial mesh ``tpm``."""

    elements = tpm.composition
    cochain = {}
    for e in elements:
        element = elements[e]
        etype = element.etype
        if etype in ("orthogonal hexahedron", ):
            cochain[e] = ___332_msepy_orthogonal___(element, cf_t, degree)
        else:
            raise NotImplementedError()
    return cochain


def ___332_raw_msepy_orthogonal___(element, cf_t, degree):
    """"""
    (coo_x, area_dydz,
     coo_y, area_dzdx,
     coo_z, area_dxdy, quad_weights) = _msepy_data_preparation(degree)

    x, y, z = element.ct.mapping(*coo_x)
    jx = element.ct.Jacobian_matrix(*coo_x)
    u, v, w = cf_t(x, y, z)
    Jx_0 = jx[1][1]*jx[2][2] - jx[1][2]*jx[2][1]
    Jx_1 = jx[2][1]*jx[0][2] - jx[2][2]*jx[0][1]
    Jx_2 = jx[0][1]*jx[1][2] - jx[0][2]*jx[1][1]
    uvw_dydz = Jx_0*u + Jx_1*v + Jx_2*w

    local_dydz = (
        np.einsum(
            'kij, ij, k -> k',
            uvw_dydz,
            np.tensordot(quad_weights[1], quad_weights[2], axes=0),
            area_dydz,
            optimize='optimal'
        )
    )

    x, y, z = element.ct.mapping(*coo_y)
    jy = element.ct.Jacobian_matrix(*coo_y)
    u, v, w = cf_t(x, y, z)
    Jy_0 = jy[1][2]*jy[2][0] - jy[1][0]*jy[2][2]
    Jy_1 = jy[2][2]*jy[0][0] - jy[2][0]*jy[0][2]
    Jy_2 = jy[0][2]*jy[1][0] - jy[0][0]*jy[1][2]
    uvw_dzdx = Jy_0*u + Jy_1*v + Jy_2*w
    local_dzdx = (
        np.einsum(
            'kij, ij, k -> k',
            uvw_dzdx,
            np.tensordot(quad_weights[0], quad_weights[2], axes=0),
            area_dzdx,
            optimize='optimal'
        )
    )

    x, y, z = element.ct.mapping(*coo_z)
    jz = element.ct.Jacobian_matrix(*coo_z)
    u, v, w = cf_t(x, y, z)
    Jz_0 = jz[1][0]*jz[2][1] - jz[1][1]*jz[2][0]
    Jz_1 = jz[2][0]*jz[0][1] - jz[2][1]*jz[0][0]
    Jz_2 = jz[0][0]*jz[1][1] - jz[0][1]*jz[1][0]
    uvw_dxdy = Jz_0*u + Jz_1*v + Jz_2*w

    local_dxdy = (
        np.einsum(
            'kij, ij, k -> k',
            uvw_dxdy,
            np.tensordot(quad_weights[0], quad_weights[1], axes=0),
            area_dxdy,
            optimize='optimal'
        )
    )

    return local_dydz, local_dzdx, local_dxdy


def ___332_msepy_orthogonal___(element, cf_t, degree):
    """"""
    local_dydz, local_dzdx, local_dxdy = ___332_raw_msepy_orthogonal___(element, cf_t, degree)
    if 'm3n3k2' in element.dof_reverse_info:
        face_indices = element.dof_reverse_info['m3n3k2']

        for fi in face_indices:
            component, local_dofs = element.find_local_dofs_on_face(
                indicator='m3n3k2', degree=degree, face_index=fi, component_wise=True
            )
            if component == 0:
                local_dydz[local_dofs] = - local_dydz[local_dofs]
            elif component == 1:
                local_dzdx[local_dofs] = - local_dzdx[local_dofs]
            elif component == 2:
                local_dxdy[local_dofs] = - local_dxdy[local_dofs]
            else:
                raise Exception()
    else:
        pass

    return np.concatenate([local_dydz, local_dzdx, local_dxdy])


_cache332_data_ = {}
from msehtt.static.mesh.great.elements.types.orthogonal_hexahedron import MseHttGreatMeshOrthogonalHexahedronElement


def _msepy_data_preparation(degree):
    """
    """

    p, btype = MseHttGreatMeshOrthogonalHexahedronElement.degree_parser(degree)
    key = str(p) + btype

    if key in _cache332_data_:
        data = _cache332_data_[key]
    else:
        quad_degree = [_ + 2 for _ in p]
        quad_nodes, quad_weights = quadrature(tuple(quad_degree), 'Gauss').quad

        num_basis_components = [
            (p[0] + 1) * p[1] * p[2],
            p[0] * (p[1] + 1) * p[2],
            p[0] * p[1] * (p[2] + 1),
        ]
        nodes = [quadrature(_, btype).quad[0] for _ in p]

        # dy dz face ________________________________________________________________________
        xi = np.zeros((num_basis_components[0], quad_degree[1] + 1, quad_degree[2] + 1))
        et = np.zeros((num_basis_components[0], quad_degree[1] + 1, quad_degree[2] + 1))
        si = np.zeros((num_basis_components[0], quad_degree[1] + 1, quad_degree[2] + 1))
        area_dydz = np.zeros((num_basis_components[0]))
        for k in range(p[2]):
            for j in range(p[1]):
                for i in range(p[0] + 1):
                    m = i + j * (p[0] + 1) + k * (p[0] + 1) * p[1]
                    xi[m, ...] = np.ones((quad_degree[1] + 1, quad_degree[2] + 1)) * nodes[0][i]
                    et[m, ...] = (quad_nodes[1][:, np.newaxis].repeat(quad_degree[2] + 1, axis=1) + 1) * (
                            nodes[1][j + 1] - nodes[1][j]) / 2 + nodes[1][j]
                    si[m, ...] = (quad_nodes[2][np.newaxis, :].repeat((quad_degree[1] + 1), axis=0) + 1) * (
                            nodes[2][k + 1] - nodes[2][k]) / 2 + nodes[2][k]
                    area_dydz[m] = (nodes[2][k + 1] - nodes[2][k]) * (nodes[1][j + 1] - nodes[1][j])
        coo_x = (xi, et, si)
        # dz dx face _________________________________________________________________________
        xi = np.zeros((num_basis_components[1], quad_degree[0] + 1, quad_degree[2] + 1))
        et = np.zeros((num_basis_components[1], quad_degree[0] + 1, quad_degree[2] + 1))
        si = np.zeros((num_basis_components[1], quad_degree[0] + 1, quad_degree[2] + 1))
        area_dzdx = np.zeros((num_basis_components[1]))
        for k in range(p[2]):
            for j in range(p[1] + 1):
                for i in range(p[0]):
                    m = i + j * p[0] + k * (p[1] + 1) * p[0]
                    xi[m, ...] = (quad_nodes[0][:, np.newaxis].repeat(quad_degree[2] + 1, axis=1) + 1) * (
                            nodes[0][i + 1] - nodes[0][i]) / 2 + nodes[0][i]
                    et[m, ...] = np.ones((quad_degree[0] + 1, quad_degree[2] + 1)) * nodes[1][j]
                    si[m, ...] = (quad_nodes[2][np.newaxis, :].repeat(quad_degree[0] + 1, axis=0) + 1) * (
                            nodes[2][k + 1] - nodes[2][k]) / 2 + nodes[2][k]
                    area_dzdx[m] = (nodes[2][k + 1] - nodes[2][k]) * (nodes[0][i + 1] - nodes[0][i])
        coo_y = (xi, et, si)
        # dx dy face _________________________________________________________________________
        xi = np.zeros((num_basis_components[2], quad_degree[0] + 1, quad_degree[1] + 1))
        et = np.zeros((num_basis_components[2], quad_degree[0] + 1, quad_degree[1] + 1))
        si = np.zeros((num_basis_components[2], quad_degree[0] + 1, quad_degree[1] + 1))
        area_dxdy = np.zeros((num_basis_components[2]))
        for k in range(p[2] + 1):
            for j in range(p[1]):
                for i in range(p[0]):
                    m = i + j * p[0] + k * p[1] * p[0]
                    xi[m, ...] = (quad_nodes[0][:, np.newaxis].repeat(quad_degree[1] + 1, axis=1) + 1) * (
                            nodes[0][i + 1] - nodes[0][i]) / 2 + nodes[0][i]
                    et[m, ...] = (quad_nodes[1][np.newaxis, :].repeat(quad_degree[0] + 1, axis=0) + 1) * (
                            nodes[1][j + 1] - nodes[1][j]) / 2 + nodes[1][j]
                    si[m, ...] = np.ones((quad_degree[0] + 1, quad_degree[1] + 1)) * nodes[2][k]
                    area_dxdy[m] = (nodes[1][j + 1] - nodes[1][j]) * (nodes[0][i + 1] - nodes[0][i])
        coo_z = (xi, et, si)
        # ===================================================================================

        data = (coo_x, area_dydz * 0.25,
                coo_y, area_dzdx * 0.25,
                coo_z, area_dxdy * 0.25, quad_weights)

        _cache332_data_[key] = data

    return data
