# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from scipy.sparse import csr_matrix

from msehtt.static.space.local_numbering.Lambda.ln_m3n3k3 import _ln_m3n3k3_msepy_quadrilateral_
from msehtt.static.space.local_numbering.Lambda.ln_m3n3k2 import _ln_m3n3k2_msepy_quadrilateral_

_cache_E332_ = {}
from src.spaces.main import _degree_str_maker


def incidence_matrix_Lambda__m3n3k2(tpm, degree):
    """"""
    key = tpm.__repr__() + 'o' + _degree_str_maker(degree)
    if key in _cache_E332_:
        return _cache_E332_[key]

    rank_elements = tpm.composition
    E = {}
    cache_key_dict = {}
    for e in rank_elements:
        element = rank_elements[e]
        etype = element.etype
        if etype in ('orthogonal hexahedron', ):
            E[e], cache_key_dict[e] = _im332_msepy_quadrilateral_(element, degree)
        else:
            raise NotImplementedError()
    _cache_E332_[key] = E, cache_key_dict
    return E, cache_key_dict


_cache_msepy_332_ = {}


def _im332_msepy_quadrilateral_(element, degree):
    """"""
    p = element.degree_parser(degree)[0]
    if p in _cache_msepy_332_:
        E, cache_key = _cache_msepy_332_[p]
    else:
        px, py, pz = p
        num_local_dof_2f = (px+1) * py * pz + px * (py+1) * pz + px * py * (pz+1)
        num_local_dof_3f = px * py * pz

        sn = _ln_m3n3k2_msepy_quadrilateral_(p)
        dn = _ln_m3n3k3_msepy_quadrilateral_(p)
        E = np.zeros(
            (
                num_local_dof_3f,
                num_local_dof_2f
            ),
            dtype=int
        )
        I, J, K = np.shape(dn)
        for k in range(K):
            for j in range(J):
                for i in range(I):
                    index0 = dn[i, j, k]
                    E[index0, sn[0][i, j, k]] = -1      # x-
                    E[index0, sn[0][i+1, j, k]] = +1    # x+
                    E[index0, sn[1][i, j, k]] = -1      # y-
                    E[index0, sn[1][i, j+1, k]] = +1    # y+
                    E[index0, sn[2][i, j, k]] = -1      # z-
                    E[index0, sn[2][i, j, k+1]] = +1    # z+

        E = csr_matrix(E)
        cache_key = f'mq{p}'
        _cache_msepy_332_[p] = E, cache_key

    dof_reverse_info = element.dof_reverse_info
    if dof_reverse_info == {}:
        return E, cache_key
    else:
        raise NotImplementedError()
