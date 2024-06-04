# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from scipy.sparse import csr_matrix

from msehtt.static.space.local_numbering.Lambda.ln_m3n3k0 import _ln_m3n3k0_msepy_quadrilateral_
from msehtt.static.space.local_numbering.Lambda.ln_m3n3k1 import _ln_m3n3k1_msepy_quadrilateral_


_cache_E330_ = {}
from src.spaces.main import _degree_str_maker


def incidence_matrix_Lambda__m3n3k0(tpm, degree):
    """"""
    key = tpm.__repr__() + 'o' + _degree_str_maker(degree)
    if key in _cache_E330_:
        return _cache_E330_[key]

    rank_elements = tpm.composition
    E = {}
    cache_key_dict = {}
    for e in rank_elements:
        element = rank_elements[e]
        etype = element.etype
        if etype in ('orthogonal hexahedron', ):
            E[e], cache_key_dict[e] = _im330_msepy_quadrilateral_(element, degree)
        else:
            raise NotImplementedError()
    _cache_E330_[key] = E, cache_key_dict
    return E, cache_key_dict


_cache_msepy_330_ = {}


def _im330_msepy_quadrilateral_(element, degree):
    """"""
    p = element.degree_parser(degree)[0]
    if p in _cache_msepy_330_:
        E, cache_key = _cache_msepy_330_[p]
    else:
        px, py, pz = p
        num_local_dof_0f = (px+1) * (py+1) * (pz+1)
        num_local_dof_1f = (
                px * (py+1) * (pz+1) +
                (px+1) * py * (pz+1) +
                (px+1) * (py+1) * pz
        )

        sn = _ln_m3n3k0_msepy_quadrilateral_(p)
        dn = _ln_m3n3k1_msepy_quadrilateral_(p)
        E = np.zeros(
            (
                num_local_dof_1f,
                num_local_dof_0f
            ),
            dtype=int
        )

        I, J, K = np.shape(dn[0])  # dx edges
        for k in range(K):
            for j in range(J):
                for i in range(I):
                    index0 = dn[0][i, j, k]
                    E[index0, sn[i, j, k]] = -1      # x-
                    E[index0, sn[i+1, j, k]] = +1    # x+

        I, J, K = np.shape(dn[1])  # dy edges
        for k in range(K):
            for j in range(J):
                for i in range(I):
                    index0 = dn[1][i, j, k]
                    E[index0, sn[i, j, k]] = -1      # y-
                    E[index0, sn[i, j+1, k]] = +1    # y+

        I, J, K = np.shape(dn[2])  # dz edges
        for k in range(K):
            for j in range(J):
                for i in range(I):
                    index0 = dn[2][i, j, k]
                    E[index0, sn[i, j, k]] = -1      # z-
                    E[index0, sn[i, j, k+1]] = +1    # z+

        E = csr_matrix(E)
        cache_key = f'mq{p}'
        _cache_msepy_330_[p] = E, cache_key

    dof_reverse_info = element.dof_reverse_info
    if dof_reverse_info == {}:
        return E, cache_key
    else:
        raise NotImplementedError()
