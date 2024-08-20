# -*- coding: utf-8 -*-
r"""
"""
from tools.quadrature import quadrature
from src.spaces.main import _degree_str_maker
import numpy as np
from scipy.sparse import csr_matrix


_cache_mm330_ = {}


def mass_matrix_Lambda__m3n3k0(tpm, degree):
    """"""
    key = tpm.__repr__() + _degree_str_maker(degree)
    if key in _cache_mm330_:
        return _cache_mm330_[key]
    M = {}
    cache_key_dict = {}
    for e in tpm.composition:
        element = tpm.composition[e]
        etype = element.etype
        if etype == 'orthogonal hexahedron':
            M[e], cache_key_dict[e] = ___mm330_orthogonal_hexahedron___(element, degree)
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    _cache_mm330_[key] = M, cache_key_dict
    return M, cache_key_dict


_cache_330_ = {}


def ___mm330_orthogonal_hexahedron___(element, degree):
    """"""
    key = element.metric_signature + _degree_str_maker(degree)
    if key in _cache_330_:
        M, cache_key = _cache_330_[key]
    else:
        p, btype = element.degree_parser(degree)
        quad_degree = (p[0], p[1], p[2])
        quad = quadrature(quad_degree, btype)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et_sg, bf = element.bf('m3n3k0', degree, *quad_nodes)
        detJM = element.ct.Jacobian(*xi_et_sg)
        M = np.einsum(
            'm, im, jm -> ij',
            quad_weights * detJM,
            bf[0], bf[0],
            optimize='optimal'
        )
        M = csr_matrix(M)
        cache_key = key
        _cache_330_[key] = M, cache_key

    return M, cache_key
