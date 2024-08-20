# -*- coding: utf-8 -*-
r"""
"""
from tools.quadrature import quadrature
from src.spaces.main import _degree_str_maker
import numpy as np
from scipy.sparse import csr_matrix

_cache_mm222_ = {}


def mass_matrix_Lambda__m2n2k2(tpm, degree):
    """"""
    key = tpm.__repr__() + _degree_str_maker(degree)
    if key in _cache_mm222_:
        return _cache_mm222_[key]
    M = {}
    cache_key_dict = {}
    for e in tpm.composition:
        element = tpm.composition[e]
        etype = element.etype
        if etype == 'orthogonal rectangle':
            M[e], cache_key_dict[e] = ___mm222_orthogonal_rectangle___(element, degree)
        elif etype == 'unique msepy curvilinear quadrilateral':
            M[e], cache_key_dict[e] = ___mm222_msepy_unique_quadrilateral___(element, degree)
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    _cache_mm222_[key] = M, cache_key_dict
    return M, cache_key_dict


_cache_222_ = {}


def ___mm222_orthogonal_rectangle___(element, degree):
    """"""
    key = element.metric_signature + _degree_str_maker(degree)
    if key in _cache_222_:
        M, cache_key = _cache_222_[key]
    else:
        p, _ = element.degree_parser(degree)
        quad_degree = (p[0], p[1])
        quad = quadrature(quad_degree, 'Gauss')
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, bf = element.bf('m2n2k2', degree, *quad_nodes)
        detJM = element.ct.Jacobian(*xi_et)
        M = np.einsum(
            'm, im, jm -> ij',
            quad_weights * np.reciprocal(detJM),
            bf[0], bf[0],
            optimize='optimal'
        )
        M = csr_matrix(M)
        cache_key = key
        _cache_222_[key] = M, cache_key

    return M, cache_key


def ___mm222_msepy_unique_quadrilateral___(element, degree):
    """"""
    p, _ = element.degree_parser(degree)

    quad_degree = (p[0]+1, p[1]+1)
    quad = quadrature(quad_degree, 'Gauss')
    quad_nodes = quad.quad_nodes
    quad_weights = quad.quad_weights_ravel
    xi_et, bf = element.bf('m2n2k2', degree, *quad_nodes)
    detJM = element.ct.Jacobian(*xi_et)
    M = np.einsum(
        'm, im, jm -> ij',
        quad_weights * np.reciprocal(detJM),
        bf[0], bf[0],
        optimize='optimal'
    )
    M = csr_matrix(M)
    cache_key = 'unique'
    return M, cache_key
