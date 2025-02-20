# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import csr_matrix

from msehtt.static.space.local_numbering.Lambda.ln_m2n2k0 import _ln_m2n2k0_msepy_quadrilateral_
from msehtt.static.space.local_numbering.Lambda.ln_m2n2k1 import _ln_m2n2k1_inner_msepy_quadrilateral_
from msehtt.static.space.local_numbering.Lambda.ln_m2n2k1 import _ln_m2n2k1_outer_msepy_quadrilateral_

from msehtt.static.space.local_numbering.Lambda.ln_m2n2k0 import _ln_m2n2k0_vtu_5_
from msehtt.static.space.local_numbering.Lambda.ln_m2n2k1 import _ln_m2n2k1_inner_vtu5_
from msehtt.static.space.local_numbering.Lambda.ln_m2n2k1 import _ln_m2n2k1_outer_vtu5_

from src.spaces.main import _degree_str_maker


_cache_E220_ = {}


# ---------------- OUTER ------------------------------------------------------


def incidence_matrix_Lambda__m2n2k0_outer(tpm, degree):
    """"""
    key = tpm.__repr__() + 'o' + _degree_str_maker(degree)
    if key in _cache_E220_:
        return _cache_E220_[key]

    rank_elements = tpm.composition
    E = {}
    cache_key_dict = {}
    for e in rank_elements:
        element = rank_elements[e]
        etype = element.etype
        if etype in ('unique msepy curvilinear quadrilateral', 'orthogonal rectangle'):
            E[e], cache_key_dict[e] = _im220o_msepy_quadrilateral_(element, degree)

        elif etype in (9, 'unique curvilinear quad'):
            E[e], cache_key_dict[e] = _im220o_vtu_9_(element, degree)

        elif etype in (5, 'unique msepy curvilinear triangle'):
            E[e], cache_key_dict[e] = _im220o_vtu_5_(element, degree)

        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    _cache_E220_[key] = E, cache_key_dict
    return E, cache_key_dict


_cache_msepy_o_ = {}


def _im220o_msepy_quadrilateral_(element, degree, check_reverse=True):
    """Outer 0-form: rot."""
    p = element.degree_parser(degree)[0]
    if p in _cache_msepy_o_:
        E, cache_key = _cache_msepy_o_[p]
    else:
        px, py = p
        num_local_dof_0f = (px+1) * (py+1)
        num_local_dof_1f = (px+1) * py + px * (py+1)

        sn = _ln_m2n2k0_msepy_quadrilateral_(p)
        dn = _ln_m2n2k1_outer_msepy_quadrilateral_(p)
        E = np.zeros(
            (
                num_local_dof_1f,
                num_local_dof_0f
            ),
            dtype=int
        )
        I, J = np.shape(dn[0])    # dy edges
        for j in range(J):
            for i in range(I):
                E[dn[0][i, j], sn[i, j]] = -1     # y-
                E[dn[0][i, j], sn[i, j+1]] = +1   # y+
        I, J = np.shape(dn[1])    # dx edges
        for j in range(J):
            for i in range(I):
                E[dn[1][i, j], sn[i, j]] = +1     # x-
                E[dn[1][i, j], sn[i+1, j]] = -1   # x+

        E = csr_matrix(E)
        cache_key = f'mq{p}'
        _cache_msepy_o_[p] = E, cache_key

    if check_reverse:
        assert element.dof_reverse_info == {}  # msepy regular element must has no reversing dofs.
    else:
        pass
    return E, cache_key


_cache_vtu9_o_ = {}


def _im220o_vtu_9_(element, degree):
    """Outer 0-form: rot."""
    E, cache_key = _im220o_msepy_quadrilateral_(element, degree, check_reverse=False)

    if 'm2n2k1_outer' not in element.dof_reverse_info:
        return E, cache_key
    else:
        face_indices = element.dof_reverse_info['m2n2k1_outer']
        new_cache_key = cache_key + str(face_indices)

        if new_cache_key in _cache_vtu9_o_:
            new_E = _cache_vtu9_o_[new_cache_key]
        else:
            new_E = E.copy()
            for fi in face_indices:
                local_dofs = element.find_local_dofs_on_face(
                    indicator='m2n2k1_outer', degree=degree, face_index=fi, component_wise=False
                )
                new_E[local_dofs, :] = -1 * E[local_dofs, :]
                new_cache_key += str(fi)
            _cache_vtu9_o_[new_cache_key] = new_E

        return new_E, new_cache_key


_cache_vtu5_o_ = {}


def _im220o_vtu_5_(element, degree):
    """Outer 0-form: rot.

    vtu 5 = 0-form local numbering

    -----------------------> et
    |
    |     0         0         0
    |     ---------------------
    |     |         |         |
    |     |         |         |
    |     |         |         |
    |   1 -----------3--------- 5
    |     |         |         |
    |     |         |         |
    |     |         |         |
    |   2 -----------4--------- 6
    |
    v
     xi

    vtu 5 = 1-form local numbering (outer)

    -----------------------> et
    |
    |
    |     ---------------------
    |     |         |         |
    |     4         6         8
    |     |         |         |
    |     -----0---------2-----
    |     |         |         |
    |     5         7         9
    |     |         |         |
    |     -----1----- ---3-----
    |
    v
     xi

    """
    p = element.degree_parser(degree)[0]
    if p in _cache_vtu5_o_:
        E, cache_key = _cache_vtu5_o_[p]
    else:
        px, py = p
        num_local_dof_0f = px * (py+1) + 1
        num_local_dof_1f = px * py + px * (py+1)

        sn = _ln_m2n2k0_vtu_5_(p)
        dn = _ln_m2n2k1_outer_vtu5_(p)
        E = np.zeros(
            (
                num_local_dof_1f,
                num_local_dof_0f
            ),
            dtype=int
        )
        I, J = np.shape(dn[0])    # dy edges
        for j in range(J):
            for i in range(I):
                E[dn[0][i, j], sn[i+1, j]] = -1     # y-
                E[dn[0][i, j], sn[i+1, j+1]] = +1   # y+
        I, J = np.shape(dn[1])    # dx edges
        for j in range(J):
            for i in range(I):
                E[dn[1][i, j], sn[i, j]] = +1     # x-
                E[dn[1][i, j], sn[i+1, j]] = -1   # x+

        E = csr_matrix(E)
        cache_key = f'vtu5o{p}'
        _cache_vtu5_o_[p] = E, cache_key

    if 'm2n2k1_outer' not in element.dof_reverse_info:
        return E, cache_key
    else:
        face_indices = element.dof_reverse_info['m2n2k1_outer']
        new_cache_key = cache_key + str(face_indices)

        if new_cache_key in _cache_vtu5_o_:
            new_E = _cache_vtu5_o_[new_cache_key]
        else:
            new_E = E.copy()
            for fi in face_indices:
                local_dofs = element.find_local_dofs_on_face(
                    indicator='m2n2k1_outer', degree=degree, face_index=fi, component_wise=False
                )
                new_E[local_dofs, :] = -1 * E[local_dofs, :]
                new_cache_key += str(fi)
            _cache_vtu5_o_[new_cache_key] = new_E

        return new_E, new_cache_key


# ---------------- INNER ------------------------------------------------------


def incidence_matrix_Lambda__m2n2k0_inner(tpm, degree):
    """"""
    key = tpm.__repr__() + 'i' + _degree_str_maker(degree)
    if key in _cache_E220_:
        return _cache_E220_[key]
    rank_elements = tpm.composition
    E = {}
    cache_key_dict = {}
    for e in rank_elements:
        element = rank_elements[e]
        etype = element.etype
        if etype in ('unique msepy curvilinear quadrilateral', 'orthogonal rectangle'):
            E[e], cache_key_dict[e] = _im220i_msepy_quadrilateral_(element, degree)

        elif etype in (9, 'unique curvilinear quad'):
            E[e], cache_key_dict[e] = _im220i_vtu_9_(element, degree)

        elif etype in (5, 'unique msepy curvilinear triangle'):
            E[e], cache_key_dict[e] = _im220i_vtu_5_(element, degree)

        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    _cache_E220_[key] = E, cache_key_dict
    return E, cache_key_dict


_cache_msepy_i_ = {}


def _im220i_msepy_quadrilateral_(element, degree, check_reverse=True):
    """Inner 0-form: grad"""
    p = element.degree_parser(degree)[0]
    if p in _cache_msepy_i_:
        E, cache_key = _cache_msepy_i_[p]
    else:
        px, py = p
        num_local_dof_0f = (px+1) * (py+1)
        num_local_dof_1f = (px+1) * py + px * (py+1)

        sn = _ln_m2n2k0_msepy_quadrilateral_(p)
        dn = _ln_m2n2k1_inner_msepy_quadrilateral_(p)
        E = np.zeros(
            (
                num_local_dof_1f,
                num_local_dof_0f
            ),
            dtype=int
        )

        I, J = np.shape(dn[0])  # dx edges
        for j in range(J):
            for i in range(I):
                E[dn[0][i, j], sn[i, j]] = -1        # x-
                E[dn[0][i, j], sn[i + 1, j]] = +1    # x+
        I, J = np.shape(dn[1])  # dy edges
        for j in range(J):
            for i in range(I):
                E[dn[1][i, j], sn[i, j]] = -1      # y-
                E[dn[1][i, j], sn[i, j + 1]] = +1  # y+

        E = csr_matrix(E)
        cache_key = f"mq{p}"
        _cache_msepy_i_[p] = E, cache_key

    if check_reverse:
        assert element.dof_reverse_info == {}   # msepy regular element must has no reversing dofs.
    else:
        pass
    return E, cache_key


_cache_vtu9_i_ = {}


def _im220i_vtu_9_(element, degree):
    """Inner 0-form: grad
    """
    E, cache_key = _im220i_msepy_quadrilateral_(element, degree, check_reverse=False)

    dof_reverse_info = element.dof_reverse_info
    if 'm2n2k1_inner' not in dof_reverse_info:
        return E, cache_key
    else:
        face_indices = dof_reverse_info['m2n2k1_inner']
        new_cache_key = cache_key + str(face_indices)

        if new_cache_key in _cache_vtu9_i_:
            new_E = _cache_vtu9_i_[new_cache_key]
        else:
            new_E = E.copy()
            for fi in face_indices:
                local_dofs = element.find_local_dofs_on_face(
                    indicator='m2n2k1_inner', degree=degree, face_index=fi, component_wise=False
                )
                new_E[local_dofs, :] = -1 * E[local_dofs, :]
            _cache_vtu9_i_[new_cache_key] = new_E

        return new_E, new_cache_key


_cache_vtu5_i_ = {}


def _im220i_vtu_5_(element, degree):
    """Inner 0-form: grad

    vtu 5 = 0-form local numbering

    -----------------------> et
    |
    |     0         0         0
    |     ---------------------
    |     |         |         |
    |     |         |         |
    |     |         |         |
    |   1 -----------3--------- 5
    |     |         |         |
    |     |         |         |
    |     |         |         |
    |   2 -----------4--------- 6
    |
    v
     xi

    vtu 5 = 1-form local numbering (inner)

    -----------------------> et
    |
    |
    |     ---------------------
    |     |         |         |
    |     0         2         4
    |     |         |         |
    |     -----6---------8-----
    |     |         |         |
    |     1         3         5
    |     |         |         |
    |     -----7----- ---9-----
    |
    v
     xi

    """
    p = element.degree_parser(degree)[0]
    if p in _cache_vtu5_i_:
        E, cache_key = _cache_vtu5_i_[p]
    else:
        px, py = p
        num_local_dof_0f = px * (py+1) + 1
        num_local_dof_1f = (px+1) * py + px * py

        sn = _ln_m2n2k0_vtu_5_(p)
        dn = _ln_m2n2k1_inner_vtu5_(p)
        E = np.zeros(
            (
                num_local_dof_1f,
                num_local_dof_0f,
            ),
            dtype=int
        )

        I, J = np.shape(dn[0])  # dx edges
        for j in range(J):
            for i in range(I):
                E[dn[0][i, j], sn[i, j]] = -1        # x-
                E[dn[0][i, j], sn[i + 1, j]] = +1    # x+
        I, J = np.shape(dn[1])  # dy edges
        for j in range(J):
            for i in range(I):
                E[dn[1][i, j], sn[i+1, j]] = -1      # y-
                E[dn[1][i, j], sn[i+1, j + 1]] = +1  # y+

        E = csr_matrix(E)
        cache_key = f"vtu5i{p}"
        _cache_vtu5_i_[p] = E, cache_key

    dof_reverse_info = element.dof_reverse_info
    if 'm2n2k1_inner' not in dof_reverse_info:
        return E, cache_key
    else:
        face_indices = dof_reverse_info['m2n2k1_inner']
        new_cache_key = cache_key + str(face_indices)

        if new_cache_key in _cache_vtu5_i_:
            new_E = _cache_vtu5_i_[new_cache_key]
        else:
            new_E = E.copy()
            for fi in face_indices:
                local_dofs = element.find_local_dofs_on_face(
                    indicator='m2n2k1_inner', degree=degree, face_index=fi, component_wise=False
                )
                new_E[local_dofs, :] = -1 * E[local_dofs, :]
            _cache_vtu5_i_[new_cache_key] = new_E

        return new_E, new_cache_key
