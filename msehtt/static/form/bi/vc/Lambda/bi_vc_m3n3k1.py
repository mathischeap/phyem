# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.quadrature import quadrature
_one_array_ = np.array([1])


def bi_vc__m3n3k1(rf1, t, vc, boundary_section):
    """ <tr-star-rf2 | tr-rf1> over boundary_section. ``vc`` represent rf2, not tr-star-rf2!

    Parameters
    ----------
    rf1
    t
    vc
    boundary_section

    Returns
    -------

    """
    assert vc.shape == (3,) and vc.ndim == 3, f"Need a vector in time + 3d-space."
    degree = rf1.degree
    tvc = vc[t]
    data_dict = {}
    for element_index__face_id in boundary_section:  # go through all rank boundary element faces
        element_index, face_id = element_index__face_id
        element = rf1.tgm.elements[element_index]
        etype = element.etype
        face = boundary_section[element_index__face_id]
        if etype in ("orthogonal hexahedron", ):
            num_test_form_local_dofs, local_dofs, face_boundary_integration_vec = (
                ___bi_vc_331_orthogonal_hexahedral___(element, degree, face, tvc))
        else:
            raise NotImplementedError()

        if element_index in data_dict:
            assert len(data_dict[element_index]) == num_test_form_local_dofs
        else:
            data_dict[element_index] = np.zeros(num_test_form_local_dofs)

        data_dict[element_index][local_dofs] = face_boundary_integration_vec

    return data_dict


from msehtt.static.space.reconstruction_matrix.Lambda.RM_m3n3k1 import ___rm331_msepy_hexahedral___
from msehtt.static.space.find.local_dofs_on_face.Lambda.m3n3k1 import __m3n3k1_msepy_hexahedral_
from msehtt.static.space.num_local_dofs.Lambda.num_local_dofs_m3n3k1 import _num_local_dofs__m3n3k1_msepy_hexahedral_


def ___bi_vc_331_orthogonal_hexahedral___(element, degree, face, tvc):
    """"""
    p = element.degree_parser(degree)[0]
    num_test_form_local_dofs = _num_local_dofs__m3n3k1_msepy_hexahedral_(p)[0]

    quad_degree = tuple([_ + 1 for _ in p])
    nodes, weights = quadrature(quad_degree, 'Gauss').quad
    face_id = face._id
    local_dofs = __m3n3k1_msepy_hexahedral_(p, face_id, component_wise=False)
    if face_id == 0:
        rm_nodes = (-_one_array_, nodes[1], nodes[2])  # x -
        weights = np.kron(weights[2], weights[1])
    elif face_id == 1:
        rm_nodes = (_one_array_, nodes[1], nodes[2])   # x +
        weights = np.kron(weights[2], weights[1])
    elif face_id == 2:
        rm_nodes = (nodes[0], -_one_array_, nodes[2])  # y -
        weights = np.kron(weights[2], weights[0])
    elif face_id == 3:
        rm_nodes = (nodes[0], _one_array_, nodes[2])   # y +
        weights = np.kron(weights[2], weights[0])
    elif face_id == 4:
        rm_nodes = (nodes[0], nodes[1], -_one_array_)  # z -
        weights = np.kron(weights[1], weights[0])
    elif face_id == 5:
        rm_nodes = (nodes[0], nodes[1], _one_array_)   # z +
        weights = np.kron(weights[1], weights[0])
    else:
        raise Exception()

    VAL = ___rm331_msepy_hexahedral___(element, degree, *rm_nodes)
    xyz = np.meshgrid(*rm_nodes, indexing='ij')
    xyz = [_.ravel('F') for _ in xyz]
    xyz = face.ct.mapping(*xyz)
    u, v, w = VAL
    u = u.T
    v = v.T
    w = w.T
    u = u[local_dofs]
    v = v[local_dofs]
    w = w[local_dofs]

    onv = face.ct.outward_unit_normal_vector(*nodes)
    nx, ny, nz = onv      # n = (nx, ny, nz)
    U, V, W = tvc(*xyz)   # u = (U, V, W)
    tr_perp_Star_vc = (   # trace-perp
        V * nz - W * ny,
        W * nx - U * nz,
        U * ny - V * nx
    )
    tr_perp_Star_vc__dot___trace_parallel_1f = tr_perp_Star_vc[0] * u + tr_perp_Star_vc[1] * v + tr_perp_Star_vc[2] * w
    area = face.area / 4
    face_boundary_integration_vec = np.sum(tr_perp_Star_vc__dot___trace_parallel_1f * weights * area, axis=1)
    return num_test_form_local_dofs, local_dofs, face_boundary_integration_vec
