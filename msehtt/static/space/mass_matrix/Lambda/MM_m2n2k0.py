# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import csr_matrix

from phyem.tools.quadrature import quadrature
from phyem.src.spaces.main import _degree_str_maker
from phyem.msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElement

from phyem.src.config import _setting

reconstructing_mass_matrices = _setting['reconstructing_mass_matrices']

_cache_mm220_ = {}

_unique_str_ = 'unique'


def mass_matrix_Lambda__m2n2k0(tpm, degree):
    """"""
    if reconstructing_mass_matrices:
        return _RMM_mass_matrix_Lambda__m2n2k0_(tpm, degree)
    else:
        pass

    key = tpm.__repr__() + _degree_str_maker(degree)
    if key in _cache_mm220_:
        return _cache_mm220_[key]
    M = {}
    cache_key_dict = {}
    for e in tpm.composition:
        element = tpm.composition[e]
        etype = element.etype
        if etype == 'orthogonal rectangle':
            M[e], cache_key_dict[e] = ___mm220_orthogonal_rectangle___(element, degree)
        elif etype == 'unique msepy curvilinear quadrilateral':
            M[e], cache_key_dict[e] = ___mm220_msepy_unique_quadrilateral___(element, degree)
        elif etype == 9:
            M[e], cache_key_dict[e] = ___mm220_quad_9___(element, degree)
        else:
            raise NotImplementedError(f"{__name__} not implemented for etype={etype}")
    _cache_mm220_[key] = M, cache_key_dict
    return M, cache_key_dict


_cache_220_ = {}


def ___mm220_orthogonal_rectangle___(element, degree):
    """"""
    key = element.metric_signature + _degree_str_maker(degree)
    if key in _cache_220_:
        M, cache_key = _cache_220_[key]
    else:
        p, btype = element.degree_parser(degree)
        quad_degree = (p[0], p[1])
        BTYPE = []
        for bt in btype:
            if bt in ('Gauss', 'Lobatto'):
                BTYPE.append(bt)
            else:
                BTYPE.append('Gauss')
        quad = quadrature(quad_degree, tuple(BTYPE))
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, bf = element.bf('m2n2k0', degree, *quad_nodes)
        detJM = element.ct.Jacobian(*xi_et)
        M = np.einsum(
            'm, im, jm -> ij',
            quad_weights * detJM,
            bf[0], bf[0],
            optimize='optimal'
        )
        M = csr_matrix(M)
        cache_key = key
        _cache_220_[key] = M, cache_key

    return M, cache_key


def ___mm220_msepy_unique_quadrilateral___(element, degree):
    """"""
    p, _ = element.degree_parser(degree)

    quad_degree = (p[0]+1, p[1]+1)
    quad = quadrature(quad_degree, 'Gauss')
    quad_nodes = quad.quad_nodes
    quad_weights = quad.quad_weights_ravel
    xi_et, bf = element.bf('m2n2k0', degree, *quad_nodes)
    detJM = element.ct.Jacobian(*xi_et)
    M = np.einsum(
        'm, im, jm -> ij',
        quad_weights * detJM,
        bf[0], bf[0],
        optimize='optimal'
    )
    M = csr_matrix(M)
    cache_key = _unique_str_
    return M, cache_key


def ___mm220_quad_9___(element, degree):
    """"""
    key = element.metric_signature + _degree_str_maker(degree)
    if key in _cache_220_:
        M, cache_key = _cache_220_[key]
    else:
        p, _ = element.degree_parser(degree)
        quad_degree = (p[0]+1, p[1]+1)
        quad = quadrature(quad_degree, 'Gauss')
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel
        xi_et, bf = element.bf('m2n2k0', degree, *quad_nodes)
        detJM = element.ct.Jacobian(*xi_et)
        M = np.einsum(
            'm, im, jm -> ij',
            quad_weights * detJM,
            bf[0], bf[0],
            optimize='optimal'
        )
        M = csr_matrix(M)
        cache_key = key
        _cache_220_[key] = M, cache_key
    return M, cache_key


# =====================================================================================
from phyem.msehtt.static.space.reconstruction_matrix.Lambda.RM_m2n2k0 import rm__m2n2k0


_cache_220r_ = {}


def _RMM_mass_matrix_Lambda__m2n2k0_(tpm, degree):
    r""""""
    p, _ = MseHttGreatMeshBaseElement.degree_parser(degree, m=2, n=2)
    p = (p[0] + 1, p[1] + 1)
    quad = quadrature(p, category='Gauss')
    quad_nodes = quad.quad_nodes
    qw_ravel = quad.quad_weights_ravel
    metric_coo = [_.ravel('F') for _ in np.meshgrid(*quad_nodes, indexing='ij')]
    RM = rm__m2n2k0(tpm, degree, *quad_nodes)[0]
    M = {}
    cache_key_dict = {}
    for e in tpm.composition:
        element = tpm.composition[e]
        cache_key = element.metric_bf_cache_key()  # return None or string
        if cache_key is None:
            cache_key = _unique_str_
        else:
            pass
        cache_key_dict[e] = cache_key
        if cache_key in _cache_220r_:
            m = _cache_220r_[cache_key]
        else:
            detJ = element.ct.Jacobian(*metric_coo)
            m = csr_matrix(
                np.einsum(
                    'wi, wj, w -> ij', RM[e], RM[e], detJ * qw_ravel, optimize='optimal'
                )
            )
            if cache_key != _unique_str_:
                _cache_220r_[cache_key] = m
            else:
                pass
        M[e] = m
    return M, cache_key_dict
