# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.quadrature import quadrature
_one_array_ = np.array([1])


# --------- OUTER ---------------------------------------------------------------------------------------

def bi_vc__m2n2k0_outer(rf0, t, vc, boundary_section):
    """ <tr-star-outer-rf1 | tr-outer-rf0> over boundary_section. ``vc`` represent rf1, not tr-star-outer-rf1!

    So, ``vc`` is a vector; `tr-star-outer-rf1` then becomes a scalar.

    Parameters
    ----------
    rf0 :
        Outer m2n2k0-form.

    t
    vc
    boundary_section

    Returns
    -------

    """
    assert vc.shape == (2,) and vc.ndim == 2, f"Need a vector in time + 2d-space."
    degree = rf0.degree
    tvc = vc[t]
    data_dict = {}
    for element_index__face_id in boundary_section:  # go through all rank boundary element faces
        element_index, face_id = element_index__face_id
        element = rf0.tgm.elements[element_index]
        etype = element.etype
        face = boundary_section[element_index__face_id]
        if etype in ("orthogonal rectangle", "unique msepy curvilinear quadrilateral"):
            num_test_form_local_dofs, local_dofs, face_boundary_integration_vec = (
                ___bi_vc_220o_msepy_quadrilateral___(element, degree, face, tvc))
        else:
            raise NotImplementedError()

        if element_index in data_dict:
            assert len(data_dict[element_index]) == num_test_form_local_dofs
        else:
            data_dict[element_index] = np.zeros(num_test_form_local_dofs)

        data_dict[element_index][local_dofs] = face_boundary_integration_vec

    return data_dict


from msehtt.static.space.reconstruction_matrix.Lambda.RM_m2n2k0 import ___rm220_msepy_quadrilateral___
from msehtt.static.space.find.local_dofs_on_face.Lambda.m2n2k0 import __m2n2k0_msepy_quadrilateral_


def ___bi_vc_220o_msepy_quadrilateral___(element, degree, face, tvc):
    """"""
    p = element.degree_parser(degree)[0]
    num_test_form_local_dofs = (p[0]+1) * (p[1]+1)
    if isinstance(p, (int, float)):
        quad_degree = int(p) + 2
    else:
        quad_degree = max(p) + 1
    nodes, weights = quadrature(quad_degree, 'Gauss').quad
    face_id = face._id

    local_dofs = __m2n2k0_msepy_quadrilateral_(p, face_id)

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

    v = ___rm220_msepy_quadrilateral___(element, degree, *rm_nodes)
    xy = face.ct.mapping(nodes)
    v = v.T
    v = v[local_dofs]

    onv = face.ct.outward_unit_normal_vector(nodes)
    nx, ny = onv
    U, V = tvc(*xy)
    trStar_vc = U * ny - V * nx
    # tr-star-outer-vc; vc is considered outer, star-outer-vc then inner, tr-star-outer-vc then means tangential
    trace_rf0 = v  # <~ | trace-outer-0f>
    JM = face.ct.Jacobian_matrix(nodes)
    Jacobian = np.sqrt(JM[0]**2 + JM[1]**2)
    face_boundary_integration_vec = np.sum(trStar_vc * trace_rf0 * weights * Jacobian, axis=1)

    return num_test_form_local_dofs, local_dofs, face_boundary_integration_vec


# --------- INNER ---------------------------------------------------------------------------------------

def bi_vc__m2n2k0_inner(rf0, t, vc, boundary_section):
    """ <tr-star-inner-rf1 | tr-inner-rf0> over boundary_section. ``vc`` represent rf1, not tr-star-inner-rf1!

    So, ``vc`` is a vector; `tr-star-inner-rf1` then becomes a scalar.

    Parameters
    ----------
    rf0 :
        Inner m2n2k0-form.

    t
    vc
    boundary_section

    Returns
    -------

    """
    assert vc.shape == (2,) and vc.ndim == 2, f"Need a vector in time + 2d-space."
    degree = rf0.degree
    tvc = vc[t]
    data_dict = {}
    for element_index__face_id in boundary_section:  # go through all rank boundary element faces
        element_index, face_id = element_index__face_id
        element = rf0.tgm.elements[element_index]
        etype = element.etype
        face = boundary_section[element_index__face_id]
        if etype in ("orthogonal rectangle", "unique msepy curvilinear quadrilateral"):
            num_test_form_local_dofs, local_dofs, face_boundary_integration_vec = (
                ___bi_vc_220i_msepy_quadrilateral___(element, degree, face, tvc))
        else:
            raise NotImplementedError()

        if element_index in data_dict:
            assert len(data_dict[element_index]) == num_test_form_local_dofs
        else:
            data_dict[element_index] = np.zeros(num_test_form_local_dofs)

        data_dict[element_index][local_dofs] = face_boundary_integration_vec

    return data_dict


def ___bi_vc_220i_msepy_quadrilateral___(element, degree, face, tvc):
    """"""
    p = element.degree_parser(degree)[0]
    num_test_form_local_dofs = (p[0]+1) * (p[1]+1)
    if isinstance(p, (int, float)):
        quad_degree = int(p) + 2
    else:
        quad_degree = max(p) + 1
    nodes, weights = quadrature(quad_degree, 'Gauss').quad
    face_id = face._id

    local_dofs = __m2n2k0_msepy_quadrilateral_(p, face_id)

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

    v = ___rm220_msepy_quadrilateral___(element, degree, *rm_nodes)
    xy = face.ct.mapping(nodes)
    v = v.T
    v = v[local_dofs]

    onv = face.ct.outward_unit_normal_vector(nodes)
    nx, ny = onv
    U, V = tvc(*xy)
    trStar_vc = U * nx + V * ny
    # tr-star-inner-vc; vc is considered inner, star-inner-vc then outer, tr-star-inner-vc then means norm-component.
    trace_rf0 = v  # <~ | trace-inner-0f>
    JM = face.ct.Jacobian_matrix(nodes)
    Jacobian = np.sqrt(JM[0]**2 + JM[1]**2)
    face_boundary_integration_vec = np.sum(trStar_vc * trace_rf0 * weights * Jacobian, axis=1)

    return num_test_form_local_dofs, local_dofs, face_boundary_integration_vec
