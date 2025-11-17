# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import bmat, csr_matrix

from phyem.tools.quadrature import quadrature
from phyem.src.spaces.main import _degree_str_maker

_cache_mm331_ = {}


def mass_matrix_Lambda__m3n3k1(tpm, degree):
    """"""
    key = tpm.__repr__() + _degree_str_maker(degree)
    if key in _cache_mm331_:
        return _cache_mm331_[key]
    M = {}
    cache_key_dict = {}
    for e in tpm.composition:
        element = tpm.composition[e]
        etype = element.etype
        if etype in (
            'orthogonal hexahedron',
        ):
            M[e], cache_key_dict[e] = ___mm331_orthogonal_hexahedron___(element, degree)
        elif etype in (
            "unique msepy curvilinear hexahedron",
        ):
            M[e], cache_key_dict[e] = ___mm331_unique_hexahedron___(element, degree)
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    _cache_mm331_[key] = M, cache_key_dict
    return M, cache_key_dict


_cache_331_ = {}


def ___mm331_orthogonal_hexahedron___(element, degree):
    """"""
    key = element.metric_signature + _degree_str_maker(degree)
    if key in _cache_331_:
        M, cache_key = _cache_331_[key]
    else:
        p, btype = element.degree_parser(degree)
        quad_degree = (p[0], p[1], p[2])
        quad = quadrature(quad_degree, btype)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et_sg, bf = element.bf('m3n3k1', degree, *quad_nodes)
        detJM = element.ct.Jacobian(*xi_et_sg)
        G = element.ct.inverse_metric_matrix(*xi_et_sg)
        M00 = np.einsum(
            'm, im, jm -> ij',
            quad_weights * detJM * G[0][0],
            bf[0], bf[0],
            optimize='optimal'
        )
        M11 = np.einsum(
            'm, im, jm -> ij',
            quad_weights * detJM * G[1][1],
            bf[1], bf[1],
            optimize='optimal'
        )
        M22 = np.einsum(
            'm, im, jm -> ij',
            quad_weights * detJM * G[2][2],
            bf[2], bf[2],
            optimize='optimal'
        )
        M00 = csr_matrix(M00)
        M11 = csr_matrix(M11)
        M22 = csr_matrix(M22)
        M = bmat(
            [
                (M00, None, None),
                (None, M11, None),
                (None, None, M22)
            ], format='csr'
        )
        cache_key = key
        _cache_331_[key] = M, cache_key

    return M, cache_key


def ___mm331_unique_hexahedron___(element, degree):
    """"""
    p, btype = element.degree_parser(degree)
    quad_degree = (p[0]+2, p[1]+2, p[2]+2)
    quad = quadrature(quad_degree, btype)
    quad_nodes = quad.quad_nodes
    quad_weights = quad.quad_weights_ravel
    xi_et_sg, bf = element.bf('m3n3k1', degree, *quad_nodes)
    detJM = element.ct.Jacobian(*xi_et_sg)
    G = element.ct.inverse_metric_matrix(*xi_et_sg)
    M00 = np.einsum(
        'm, im, jm -> ij',
        quad_weights * detJM * G[0][0],
        bf[0], bf[0],
        optimize='optimal'
    )
    M11 = np.einsum(
        'm, im, jm -> ij',
        quad_weights * detJM * G[1][1],
        bf[1], bf[1],
        optimize='optimal'
    )
    M22 = np.einsum(
        'm, im, jm -> ij',
        quad_weights * detJM * G[2][2],
        bf[2], bf[2],
        optimize='optimal'
    )
    M00 = csr_matrix(M00)
    M11 = csr_matrix(M11)
    M22 = csr_matrix(M22)
    M = bmat(
        [
            (M00, None, None),
            (None, M11, None),
            (None, None, M22)
        ], format='csr'
    )

    return M, 'unique'
