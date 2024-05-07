# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from scipy.sparse import csr_matrix

from msehtt.static.space.local_numbering.Lambda.ln_m2n2k0 import _ln_m2n2k0_msepy_quadrilateral_
from msehtt.static.space.local_numbering.Lambda.ln_m2n2k1 import _ln_m2n2k1_inner_msepy_quadrilateral_
from msehtt.static.space.local_numbering.Lambda.ln_m2n2k1 import _ln_m2n2k1_outer_msepy_quadrilateral_


_cache_E220_ = {}
from src.spaces.main import _degree_str_maker


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
        else:
            raise NotImplementedError()
    _cache_E220_[key] = E, cache_key_dict
    return E, cache_key_dict


_cache_msepy_o_ = {}


def _im220o_msepy_quadrilateral_(element, degree):
    """"""
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

    dof_reverse_info = element.dof_reverse_info
    if dof_reverse_info == {}:
        return E, cache_key
    else:
        raise NotImplementedError()


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
        else:
            raise NotImplementedError()
    _cache_E220_[key] = E, cache_key_dict
    return E, cache_key_dict


_cache_msepy_i_ = {}


def _im220i_msepy_quadrilateral_(element, degree):
    """"""
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

    dof_reverse_info = element.dof_reverse_info
    if dof_reverse_info == {}:
        return E, cache_key
    else:
        raise NotImplementedError()
