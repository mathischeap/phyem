# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import csr_matrix

from phyem.tools.quadrature import quadrature
from phyem.src.spaces.main import _degree_str_maker


_cache_mm333_ = {}


def mass_matrix_Lambda__m3n3k3(tpm, degree):
    """"""
    key = tpm.__repr__() + _degree_str_maker(degree)
    if key in _cache_mm333_:
        return _cache_mm333_[key]
    M = {}
    cache_key_dict = {}
    for e in tpm.composition:
        element = tpm.composition[e]
        etype = element.etype
        if etype in (
            'orthogonal hexahedron',
        ):
            M[e], cache_key_dict[e] = ___mm333_orthogonal_hexahedron___(element, degree)
        elif etype in (
            'unique msepy curvilinear hexahedron',
        ):
            M[e], cache_key_dict[e] = ___mm333_unique_hexahedron___(element, degree)
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    _cache_mm333_[key] = M, cache_key_dict
    return M, cache_key_dict


_cache_333_ = {}


def ___mm333_orthogonal_hexahedron___(element, degree):
    """"""
    key = element.metric_signature + _degree_str_maker(degree)
    if key in _cache_333_:
        M, cache_key = _cache_333_[key]
    else:
        p, _ = element.degree_parser(degree)
        quad_degree = (p[0], p[1], p[2])
        quad = quadrature(quad_degree, 'Gauss')
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et_sg, bf = element.bf('m3n3k3', degree, *quad_nodes)
        detJM = element.ct.Jacobian(*xi_et_sg)
        M = np.einsum(
            'm, im, jm -> ij',
            quad_weights * np.reciprocal(detJM),
            bf[0], bf[0],
            optimize='optimal'
        )
        M = csr_matrix(M)
        cache_key = key
        _cache_333_[key] = M, cache_key

    return M, cache_key


def ___mm333_unique_hexahedron___(element, degree):
    """"""
    p, _ = element.degree_parser(degree)
    quad_degree = (p[0], p[1], p[2])
    quad = quadrature(quad_degree, 'Gauss')
    quad_nodes = quad.quad_nodes
    quad_weights = quad.quad_weights_ravel
    xi_et_sg, bf = element.bf('m3n3k3', degree, *quad_nodes)
    detJM = element.ct.Jacobian(*xi_et_sg)
    M = np.einsum(
        'm, im, jm -> ij',
        quad_weights * np.reciprocal(detJM),
        bf[0], bf[0],
        optimize='optimal'
    )
    M = csr_matrix(M)

    return M, 'unique'
