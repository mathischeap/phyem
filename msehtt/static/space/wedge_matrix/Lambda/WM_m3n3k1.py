# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import bmat, csr_matrix

from phyem.tools.quadrature import quadrature
from phyem.src.spaces.main import _degree_str_maker


_cache_wm331_332_ = {}


def wedge_matrix_Lambda__m3n3k1_w_m3n3k2(tpm, degree0, degree1):
    r""""""
    key = tpm.__repr__() + _degree_str_maker(degree0) + '<D>' + _degree_str_maker(degree1)
    if key in _cache_wm331_332_:
        return _cache_wm331_332_[key]

    W = {}
    cache_key_dict = {}
    for e in tpm.composition:
        element = tpm.composition[e]
        etype = element.etype
        if etype in (
            'orthogonal hexahedron',
        ):
            W[e], cache_key_dict[e] = ___wm331_332_orthogonal_hexahedron___(element, degree0, degree1)

        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")

    _cache_wm331_332_[key] = W, cache_key_dict
    return W, cache_key_dict


_cache_331_332_ = {}


def ___wm331_332_orthogonal_hexahedron___(element, degree0, degree1):
    r""""""
    key = _degree_str_maker(degree0) + ' w ' + _degree_str_maker(degree1)
    if key in _cache_331_332_:
        W, cache_key = _cache_331_332_[key]
    else:
        p0, _ = element.degree_parser(degree0)
        p1, _ = element.degree_parser(degree1)
        quad_degree = (max([p0[0], p1[0]]), max([p0[1], p1[1]]), max([p0[2], p1[2]]))
        quad = quadrature(quad_degree, 'Gauss')
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        _, bf0 = element.bf('m3n3k1', degree0, *quad_nodes)
        _, bf1 = element.bf('m3n3k2', degree1, *quad_nodes)
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
        W22 = np.einsum(
            'm, im, jm -> ij',
            quad_weights,
            bf0[2], bf1[2],
            optimize='optimal'
        )
        W00 = csr_matrix(W00)
        W11 = csr_matrix(W11)
        W22 = csr_matrix(W22)
        W = bmat(
            [
                (W00, None, None),
                (None, W11, None),
                (None, None, W22)
            ], format='csr'
        )
        W = csr_matrix(W)
        cache_key = key
        _cache_331_332_[key] = W, cache_key

    return W, cache_key
