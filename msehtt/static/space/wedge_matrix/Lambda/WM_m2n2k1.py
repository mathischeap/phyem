# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import csr_matrix, bmat

from phyem.tools.quadrature import quadrature
from phyem.src.spaces.main import _degree_str_maker


# ==============================================================================================#


_cache_wm221o_221i_ = {}


def wedge_matrix_Lambda__m2n2k1_outer_w_m2n2k1_inner(tpm, degree0, degree1):
    r""""""
    key = tpm.__repr__() + _degree_str_maker(degree0) + '<D>' + _degree_str_maker(degree1)
    if key in _cache_wm221o_221i_:
        return _cache_wm221o_221i_[key]

    W = {}
    cache_key_dict = {}
    for e in tpm.composition:
        element = tpm.composition[e]
        etype = element.etype
        if etype in (
            'orthogonal rectangle',
            'unique msepy curvilinear quadrilateral',
            9,
        ):
            W[e], cache_key_dict[e] = ___wm221o_221i_orthogonal_rectangle___(element, degree0, degree1)

        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")

    _cache_wm221o_221i_[key] = W, cache_key_dict
    return W, cache_key_dict


_cache_221o_221i_ = {}


def ___wm221o_221i_orthogonal_rectangle___(element, degree0, degree1):
    r""""""
    key = _degree_str_maker(degree0) + ' w ' + _degree_str_maker(degree1)
    if key in _cache_221o_221i_:
        W, cache_key = _cache_221o_221i_[key]
    else:
        p0, _ = element.degree_parser(degree0)
        p1, _ = element.degree_parser(degree1)
        quad_degree = (max([p0[0], p1[0]]), max([p0[1], p1[1]]))
        quad = quadrature(quad_degree, 'Gauss')
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        _, bf0 = element.bf('m2n2k1_outer', degree0, *quad_nodes)
        _, bf1 = element.bf('m2n2k1_inner', degree1, *quad_nodes)
        W00 = np.einsum(
            'm, im, jm -> ij',
            quad_weights,
            bf0[0], bf1[0],
            optimize='optimal'
        )
        W11 = np.einsum(
            'm, im, jm -> ij',
            quad_weights,
            bf0[1], bf1[1],
            optimize='optimal'
        )
        W00 = csr_matrix(W00)
        W11 = csr_matrix(W11)
        W = bmat(
            [
                (W00, None),
                (None, W11)
            ], format='csr'
        )
        W = csr_matrix(W)
        cache_key = key
        _cache_221o_221i_[key] = W, cache_key

    return W, cache_key


# ==============================================================================================#


_cache_wm221i_221o_ = {}


def wedge_matrix_Lambda__m2n2k1_inner_w_m2n2k1_outer(tpm, degree0, degree1):
    r""""""
    key = tpm.__repr__() + _degree_str_maker(degree0) + '<D>' + _degree_str_maker(degree1)
    if key in _cache_wm221i_221o_:
        return _cache_wm221i_221o_[key]

    W = {}
    cache_key_dict = {}
    for e in tpm.composition:
        element = tpm.composition[e]
        etype = element.etype
        if etype in (
            'orthogonal rectangle',
            'unique msepy curvilinear quadrilateral',
            9,
        ):
            W[e], cache_key_dict[e] = ___wm221i_221o_orthogonal_rectangle___(element, degree0, degree1)

        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")

    _cache_wm221i_221o_[key] = W, cache_key_dict
    return W, cache_key_dict


_cache_221i_221o_ = {}


def ___wm221i_221o_orthogonal_rectangle___(element, degree0, degree1):
    r""""""
    key = _degree_str_maker(degree0) + ' w ' + _degree_str_maker(degree1)
    if key in _cache_221i_221o_:
        W, cache_key = _cache_221i_221o_[key]
    else:
        p0, _ = element.degree_parser(degree0)
        p1, _ = element.degree_parser(degree1)
        quad_degree = (max([p0[0], p1[0]]), max([p0[1], p1[1]]))
        quad = quadrature(quad_degree, 'Gauss')
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        _, bf0 = element.bf('m2n2k1_inner', degree0, *quad_nodes)
        _, bf1 = element.bf('m2n2k1_outer', degree1, *quad_nodes)
        W00 = np.einsum(
            'm, im, jm -> ij',
            quad_weights,
            bf0[0], bf1[0],
            optimize='optimal'
        )
        W11 = np.einsum(
            'm, im, jm -> ij',
            quad_weights,
            bf0[1], bf1[1],
            optimize='optimal'
        )
        W00 = csr_matrix(W00)
        W11 = csr_matrix(W11)
        W = bmat(
            [
                (W00, None),
                (None, W11)
            ], format='csr'
        )
        W = csr_matrix(W)
        cache_key = key
        _cache_221i_221o_[key] = W, cache_key

    return W, cache_key
