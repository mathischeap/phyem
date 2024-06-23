# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.quadrature import quadrature
_one_array_ = np.array([1])


# ------------------------- OUTER ------------------------------------------------------------------------

def bi_vc__m2n2k1_outer(rf1, t, vc, boundary_section):
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
    assert vc.shape == (1,) and vc.ndim == 2, f"Need a scalar in time + 2d-space."
    degree = rf1.degree
    tvc = vc[t]
    data_dict = {}
    for element_index__face_id in boundary_section:  # go through all rank boundary element faces
        element_index, face_id = element_index__face_id
        element = rf1.tgm.elements[element_index]
        etype = element.etype
        face = boundary_section[element_index__face_id]
        if etype in ("orthogonal rectangle", "unique msepy curvilinear quadrilateral"):
            num_test_form_local_dofs, local_dofs, face_boundary_integration_vec = (
                ___bi_vc_221o_msepy_quadrilateral___(element, degree, face, tvc))
        else:
            raise NotImplementedError()

        if element_index in data_dict:
            assert len(data_dict[element_index]) == num_test_form_local_dofs
        else:
            data_dict[element_index] = np.zeros(num_test_form_local_dofs)

        data_dict[element_index][local_dofs] = face_boundary_integration_vec

    return data_dict


from msehtt.static.space.reconstruction_matrix.Lambda.RM_m2n2k1 import ___rm221o_msepy_quadrilateral___
from msehtt.static.space.find.local_dofs_on_face.Lambda.m2n2k1 import __m2n2k1_outer_msepy_quadrilateral_


def ___bi_vc_221o_msepy_quadrilateral___(element, degree, face, tvc):
    """"""
    p = element.degree_parser(degree)[0]
    num_test_form_local_dofs = (p[0]+1) * p[1] + p[0] * (p[1]+1)
    if isinstance(p, (int, float)):
        quad_degree = int(p) + 2
    else:
        quad_degree = max(p) + 1
    nodes, weights = quadrature(quad_degree, 'Gauss').quad
    face_id = face._id

    local_dofs = __m2n2k1_outer_msepy_quadrilateral_(p, face_id, component_wise=False)

    if face_id == 0:
        rm_nodes = (-_one_array_, nodes)  # x -
    elif face_id == 1:
        rm_nodes = (_one_array_, nodes)   # x +
    elif face_id == 2:
        rm_nodes = (nodes, -_one_array_)  # y -
    elif face_id == 3:
        rm_nodes = (nodes, _one_array_)   # y +
    else:
        raise Exception()

    v = ___rm221o_msepy_quadrilateral___(element, degree, *rm_nodes)
    xy = face.ct.mapping(nodes)
    vx, vy = v
    vx = vx.T
    vy = vy.T
    vx = vx[local_dofs]
    vy = vy[local_dofs]

    onv = face.ct.outward_unit_normal_vector(nodes)
    nx, ny = onv
    trStar_vc = tvc(*xy)[0]
    trace_1f = vx * nx + vy * ny  # <~ | trace-outer-f>
    JM = face.ct.Jacobian_matrix(nodes)
    Jacobian = np.sqrt(JM[0]**2 + JM[1]**2)
    face_boundary_integration_vec = np.sum(trStar_vc * trace_1f * weights * Jacobian, axis=1)

    return num_test_form_local_dofs, local_dofs, face_boundary_integration_vec


# ------------------------- INNER ------------------------------------------------------------------------

def bi_vc__m2n2k1_inner(rf1, t, vc, boundary_section):
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
    assert vc.shape == (1,) and vc.ndim == 2, f"Need a scalar in time + 2d-space."
    degree = rf1.degree
    tvc = vc[t]
    data_dict = {}
    for element_index__face_id in boundary_section:  # go through all rank boundary element faces
        element_index, face_id = element_index__face_id
        element = rf1.tgm.elements[element_index]
        etype = element.etype
        face = boundary_section[element_index__face_id]
        if etype in ("orthogonal rectangle", "unique msepy curvilinear quadrilateral"):
            num_test_form_local_dofs, local_dofs, face_boundary_integration_vec = (
                ___bi_vc_221i_msepy_quadrilateral___(element, degree, face, tvc))
        else:
            raise NotImplementedError()

        if element_index in data_dict:
            assert len(data_dict[element_index]) == num_test_form_local_dofs
        else:
            data_dict[element_index] = np.zeros(num_test_form_local_dofs)

        data_dict[element_index][local_dofs] = face_boundary_integration_vec

    return data_dict


from msehtt.static.space.reconstruction_matrix.Lambda.RM_m2n2k1 import ___rm221i_msepy_quadrilateral___
from msehtt.static.space.find.local_dofs_on_face.Lambda.m2n2k1 import __m2n2k1_inner_msepy_quadrilateral_


def ___bi_vc_221i_msepy_quadrilateral___(element, degree, face, tvc):
    """"""
    p = element.degree_parser(degree)[0]
    num_test_form_local_dofs = (p[0]+1) * p[1] + p[0] * (p[1]+1)
    if isinstance(p, (int, float)):
        quad_degree = int(p) + 2
    else:
        quad_degree = max(p) + 1
    nodes, weights = quadrature(quad_degree, 'Gauss').quad
    face_id = face._id

    local_dofs = __m2n2k1_inner_msepy_quadrilateral_(p, face_id, component_wise=False)

    if face_id == 0:
        rm_nodes = (-_one_array_, nodes)  # x -
    elif face_id == 1:
        rm_nodes = (_one_array_, nodes)   # x +
    elif face_id == 2:
        rm_nodes = (nodes, -_one_array_)  # y -
    elif face_id == 3:
        rm_nodes = (nodes, _one_array_)   # y +
    else:
        raise Exception()

    v = ___rm221i_msepy_quadrilateral___(element, degree, *rm_nodes)
    xy = face.ct.mapping(nodes)
    vx, vy = v
    vx = vx.T
    vy = vy.T
    vx = vx[local_dofs]
    vy = vy[local_dofs]

    onv = face.ct.outward_unit_normal_vector(nodes)
    nx, ny = onv
    trStar_vc = tvc(*xy)[0]
    trace_1f = vx * ny - vy * nx  # <~ | trace-inner-f>;
    JM = face.ct.Jacobian_matrix(nodes)
    Jacobian = np.sqrt(JM[0]**2 + JM[1]**2)
    face_boundary_integration_vec = np.sum(trStar_vc * trace_1f * weights * Jacobian, axis=1)

    return num_test_form_local_dofs, local_dofs, face_boundary_integration_vec
