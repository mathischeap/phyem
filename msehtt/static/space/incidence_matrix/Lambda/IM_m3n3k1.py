# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import csr_matrix

from phyem.msehtt.static.space.local_numbering.Lambda.ln_m3n3k1 import _ln_m3n3k1_msepy_quadrilateral_
from phyem.msehtt.static.space.local_numbering.Lambda.ln_m3n3k2 import _ln_m3n3k2_msepy_quadrilateral_
from phyem.src.spaces.main import _degree_str_maker


_cache_E331_ = {}


def incidence_matrix_Lambda__m3n3k1(tpm, degree):
    """"""
    key = tpm.__repr__() + 'o' + _degree_str_maker(degree)
    if key in _cache_E331_:
        return _cache_E331_[key]

    rank_elements = tpm.composition
    E = {}
    cache_key_dict = {}
    for e in rank_elements:
        element = rank_elements[e]
        etype = element.etype
        if etype in (
            'orthogonal hexahedron',
            "unique msepy curvilinear hexahedron",
        ):
            E[e], cache_key_dict[e] = _im331_msepy_quadrilateral_(element, degree)
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    _cache_E331_[key] = E, cache_key_dict
    return E, cache_key_dict


_cache_msepy_331_ = {}


def _im331_msepy_quadrilateral_(element, degree):
    """"""
    p = element.degree_parser(degree)[0]
    if p in _cache_msepy_331_:
        E, cache_key = _cache_msepy_331_[p]
    else:
        px, py, pz = p
        num_local_dof_1f = (
                px * (py+1) * (pz+1) +
                (px+1) * py * (pz+1) +
                (px+1) * (py+1) * pz
        )
        num_local_dof_2f = (
                (px+1) * py * pz +
                px * (py+1) * pz +
                px * py * (pz+1)
        )

        sn = _ln_m3n3k1_msepy_quadrilateral_(p)
        dn = _ln_m3n3k2_msepy_quadrilateral_(p)
        E = np.zeros(
            (
                num_local_dof_2f,
                num_local_dof_1f
            ),
            dtype=int
        )

        I, J, K = np.shape(dn[0])  # dx edges
        for k in range(K):
            for j in range(J):
                for i in range(I):
                    index0 = dn[0][i, j, k]
                    E[index0, sn[1][i, j, k]] = +1
                    E[index0, sn[1][i, j, k+1]] = -1
                    E[index0, sn[2][i, j, k]] = -1
                    E[index0, sn[2][i, j+1, k]] = +1

        I, J, K = np.shape(dn[1])  # dy edges
        for k in range(K):
            for j in range(J):
                for i in range(I):
                    index0 = dn[1][i, j, k]
                    E[index0, sn[0][i, j, k]] = -1
                    E[index0, sn[0][i, j, k+1]] = +1
                    E[index0, sn[2][i, j, k]] = +1
                    E[index0, sn[2][i+1, j, k]] = -1

        I, J, K = np.shape(dn[2])  # dz edges
        for k in range(K):
            for j in range(J):
                for i in range(I):
                    index0 = dn[2][i, j, k]
                    E[index0, sn[0][i, j, k]] = +1    # x-
                    E[index0, sn[0][i, j+1, k]] = -1    # x+
                    E[index0, sn[1][i, j, k]] = -1    # x-
                    E[index0, sn[1][i+1, j, k]] = +1    # x+

        E = csr_matrix(E)
        cache_key = f'mq{p}'
        _cache_msepy_331_[p] = E, cache_key

    dof_reverse_info = element.dof_reverse_info
    if dof_reverse_info == {}:
        return E, cache_key
    else:
        raise NotImplementedError()
