# -*- coding: utf-8 -*-
r"""
"""
from tools.quadrature import quadrature
from src.spaces.main import _degree_str_maker
import numpy as np
from scipy.sparse import bmat, csr_matrix
_unique_str_ = 'unique'


# ================= OUTER =============================================================


_cache_mm221o_ = {}


def mass_matrix_Lambda__m2n2k1_outer(tpm, degree):
    r""""""
    key = tpm.__repr__() + _degree_str_maker(degree)
    if key in _cache_mm221o_:
        return _cache_mm221o_[key]
    M = {}
    cache_key_dict = {}
    for e in tpm.composition:
        element = tpm.composition[e]
        etype = element.etype
        if etype == 'orthogonal rectangle':
            M[e], cache_key_dict[e] = ___mm221o_orthogonal_rectangle___(element, degree)
        elif etype == 'unique msepy curvilinear quadrilateral':
            M[e], cache_key_dict[e] = ___mm221o_msepy_unique_quadrilateral___(element, degree)
        elif etype == 9:
            M[e], cache_key_dict[e] = ___mm221o_quad_9___(element, degree)
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    _cache_mm221o_[key] = M, cache_key_dict
    return M, cache_key_dict


_cache_221o_ = {}


def ___mm221o_orthogonal_rectangle___(element, degree):
    r""""""
    key = element.metric_signature + _degree_str_maker(degree)
    if key in _cache_221o_:
        M, cache_key = _cache_221o_[key]
    else:
        p, btype = element.degree_parser(degree)
        quad_degree = (p[0], p[1])
        quad = quadrature(quad_degree, btype)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, bf = element.bf('m2n2k1_outer', degree, *quad_nodes)
        detJM = element.ct.Jacobian(*xi_et)
        G = element.ct.inverse_metric_matrix(*xi_et)
        M00 = np.einsum(
            'm, im, jm -> ij',
            quad_weights * detJM * G[1][1],
            bf[0], bf[0],
            optimize='optimal'
        )
        M11 = np.einsum(
            'm, im, jm -> ij',
            quad_weights * detJM * G[0][0],
            bf[1], bf[1],
            optimize='optimal'
        )
        M01 = None
        M10 = None
        M00 = csr_matrix(M00)
        M11 = csr_matrix(M11)
        M = bmat(
            [
                (M00, M01),
                (M10, M11)
            ], format='csr'
        )
        cache_key = key
        _cache_221o_[key] = M, cache_key

    return M, cache_key


def ___mm221o_msepy_unique_quadrilateral___(element, degree):
    r""""""
    p, _ = element.degree_parser(degree)

    quad_degree = (p[0]+1, p[1]+1)
    quad = quadrature(quad_degree, 'Gauss')
    quad_nodes = quad.quad_nodes
    quad_weights = quad.quad_weights_ravel

    xi_et, bf = element.bf('m2n2k1_outer', degree, *quad_nodes)
    detJM = element.ct.Jacobian(*xi_et)
    G = element.ct.inverse_metric_matrix(*xi_et)

    M00 = np.einsum(
        'm, im, jm -> ij',
        quad_weights * detJM * G[1][1],
        bf[0], bf[0],
        optimize='optimal'
    )

    M11 = np.einsum(
        'm, im, jm -> ij',
        quad_weights * detJM * G[0][0],
        bf[1], bf[1],
        optimize='optimal'
    )

    M01 = - np.einsum(
        'm, im, jm -> ij',
        quad_weights * detJM * G[1][0],
        bf[0], bf[1],
        optimize='optimal'
    )
    M10 = M01.T

    M00 = csr_matrix(M00)
    M01 = csr_matrix(M01)
    M10 = csr_matrix(M10)
    M11 = csr_matrix(M11)

    M = bmat(
        [
            (M00, M01),
            (M10, M11)
        ], format='csr'
    )

    return M, _unique_str_


def ___mm221o_quad_9___(element, degree):
    r""""""
    reverse_info = element.dof_reverse_info
    if 'm2n2k1_outer' in reverse_info:
        reverse_key = str(reverse_info['m2n2k1_outer'])
    else:
        reverse_key = ''

    key = element.metric_signature + reverse_key + _degree_str_maker(degree)
    if key in _cache_221o_:
        M, cache_key = _cache_221o_[key]
    else:
        p, _ = element.degree_parser(degree)

        quad_degree = (p[0]+1, p[1]+1)
        quad = quadrature(quad_degree, 'Gauss')
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel

        xi_et, bf = element.bf('m2n2k1_outer', degree, *quad_nodes)
        detJM = element.ct.Jacobian(*xi_et)
        G = element.ct.inverse_metric_matrix(*xi_et)

        M00 = np.einsum(
            'm, im, jm -> ij',
            quad_weights * detJM * G[1][1],
            bf[0], bf[0],
            optimize='optimal'
        )

        M11 = np.einsum(
            'm, im, jm -> ij',
            quad_weights * detJM * G[0][0],
            bf[1], bf[1],
            optimize='optimal'
        )

        M01 = - np.einsum(
            'm, im, jm -> ij',
            quad_weights * detJM * G[1][0],
            bf[0], bf[1],
            optimize='optimal'
        )
        M10 = M01.T

        M00 = csr_matrix(M00)
        M01 = csr_matrix(M01)
        M10 = csr_matrix(M10)
        M11 = csr_matrix(M11)

        M = bmat(
            [
                (M00, M01),
                (M10, M11)
            ], format='csr'
        )

        cache_key = key
        _cache_221o_[key] = M, cache_key

    return M, cache_key


# ================= INNER =============================================================


_cache_mm221i_ = {}


def mass_matrix_Lambda__m2n2k1_inner(tpm, degree):
    r""""""
    key = tpm.__repr__() + _degree_str_maker(degree)
    if key in _cache_mm221i_:
        return _cache_mm221i_[key]
    M = {}
    cache_key_dict = {}
    for e in tpm.composition:
        element = tpm.composition[e]
        etype = element.etype
        if etype == 'orthogonal rectangle':
            M[e], cache_key_dict[e] = ___mm221i_orthogonal_rectangle___(element, degree)
        elif etype == 'unique msepy curvilinear quadrilateral':
            M[e], cache_key_dict[e] = ___mm221i_msepy_unique_quadrilateral___(element, degree)
        elif etype == 9:
            M[e], cache_key_dict[e] = ___mm221i_quad_9___(element, degree)
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    _cache_mm221i_[key] = M, cache_key_dict
    return M, cache_key_dict


_cache_221i_ = {}


def ___mm221i_orthogonal_rectangle___(element, degree):
    r""""""
    key = element.metric_signature + _degree_str_maker(degree)
    if key in _cache_221i_:
        M, cache_key = _cache_221i_[key]
    else:
        p, btype = element.degree_parser(degree)
        quad_degree = (p[0], p[1])
        quad = quadrature(quad_degree, btype)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, bf = element.bf('m2n2k1_inner', degree, *quad_nodes)
        detJM = element.ct.Jacobian(*xi_et)
        G = element.ct.inverse_metric_matrix(*xi_et)
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
        M01 = None
        M10 = None
        M00 = csr_matrix(M00)
        M11 = csr_matrix(M11)
        M = bmat(
            [
                (M00, M01),
                (M10, M11)
            ], format='csr'
        )
        cache_key = key
        _cache_221i_[key] = M, cache_key

    return M, cache_key


def ___mm221i_msepy_unique_quadrilateral___(element, degree):
    r""""""
    p, _ = element.degree_parser(degree)

    quad_degree = (p[0]+1, p[1]+1)
    quad = quadrature(quad_degree, 'Gauss')
    quad_nodes = quad.quad_nodes
    quad_weights = quad.quad_weights_ravel

    xi_et, bf = element.bf('m2n2k1_inner', degree, *quad_nodes)
    detJM = element.ct.Jacobian(*xi_et)
    G = element.ct.inverse_metric_matrix(*xi_et)

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

    M01 = np.einsum(
        'm, im, jm -> ij',
        quad_weights * detJM * G[0][1],
        bf[0], bf[1],
        optimize='optimal'
    )
    M10 = M01.T

    M00 = csr_matrix(M00)
    M01 = csr_matrix(M01)
    M10 = csr_matrix(M10)
    M11 = csr_matrix(M11)

    M = bmat(
        [
            (M00, M01),
            (M10, M11)
        ], format='csr'
    )

    return M, _unique_str_


def ___mm221i_quad_9___(element, degree):
    r""""""
    reverse_info = element.dof_reverse_info
    if 'm2n2k1_inner' in reverse_info:
        reverse_key = str(reverse_info['m2n2k1_inner'])
    else:
        reverse_key = ''

    key = element.metric_signature + reverse_key + _degree_str_maker(degree)
    if key in _cache_221i_:
        M, cache_key = _cache_221i_[key]
    else:
        p, _ = element.degree_parser(degree)

        quad_degree = (p[0]+1, p[1]+1)
        quad = quadrature(quad_degree, 'Gauss')
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel

        xi_et, bf = element.bf('m2n2k1_inner', degree, *quad_nodes)
        detJM = element.ct.Jacobian(*xi_et)
        G = element.ct.inverse_metric_matrix(*xi_et)

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

        M01 = np.einsum(
            'm, im, jm -> ij',
            quad_weights * detJM * G[0][1],
            bf[0], bf[1],
            optimize='optimal'
        )
        M10 = M01.T

        M00 = csr_matrix(M00)
        M01 = csr_matrix(M01)
        M10 = csr_matrix(M10)
        M11 = csr_matrix(M11)

        M = bmat(
            [
                (M00, M01),
                (M10, M11)
            ], format='csr'
        )
        cache_key = key
        _cache_221i_[key] = M, cache_key

    return M, cache_key
