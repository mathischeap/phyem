# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from msepy.tools.polynomials import Lobatto_polynomials_of_degree
from tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer


def basis_function_Lambda__m2n2k1_outer(etype, degree, xi_1d, eta_1d):
    """"""
    if etype in ('unique msepy curvilinear quadrilateral', 'orthogonal rectangle'):
        local_numbering = ___bf221o_outer_msepy_quadrilateral___(degree, xi_1d, eta_1d)
    else:
        raise NotImplementedError()
    return local_numbering


_cache_bf221o_mq_ = {}


def ___bf221o_outer_msepy_quadrilateral___(degree, xi_1d, eta_1d):
    """"""
    if isinstance(degree, int):
        p = (degree, degree)
        btype = 'Lobatto'
    else:
        raise NotImplementedError()

    key = str(p[0]) + '-' + str(p[1]) + '-' + btype
    cached, data = ndarray_key_comparer(_cache_bf221o_mq_, [xi_1d, eta_1d], check_str=key)
    if cached:
        return data
    else:
        pass

    if btype == 'Lobatto':
        bfs = (
            Lobatto_polynomials_of_degree(p[0]),
            Lobatto_polynomials_of_degree(p[1]),
        )
    else:
        raise NotImplementedError()

    xi, eta = np.meshgrid(xi_1d, eta_1d, indexing='ij')
    mesh_grid = (xi.ravel('F'), eta.ravel('F'))
    lb_xi = bfs[0].node_basis(x=xi_1d)
    ed_et = bfs[1].edge_basis(x=eta_1d)
    bf_edge_det = np.kron(ed_et, lb_xi)
    ed_xi = bfs[0].edge_basis(x=xi_1d)
    lb_et = bfs[1].node_basis(x=eta_1d)
    bf_edge_dxi = np.kron(lb_et, ed_xi)
    _basis_ = (bf_edge_det, bf_edge_dxi)
    data = mesh_grid, _basis_
    add_to_ndarray_cache(_cache_bf221o_mq_, [xi_1d, eta_1d], data, check_str=key)
    return data


def basis_function_Lambda__m2n2k1_inner(etype, degree, xi_1d, eta_1d):
    """"""
    if etype in ('unique msepy curvilinear quadrilateral', 'orthogonal rectangle'):
        local_numbering = ___bf221i_inner_msepy_quadrilateral___(degree, xi_1d, eta_1d)
    else:
        raise NotImplementedError()
    return local_numbering


_cache_bf221i_mq_ = {}


def ___bf221i_inner_msepy_quadrilateral___(degree, xi_1d, eta_1d):
    """"""
    if isinstance(degree, int):
        p = (degree, degree)
        btype = 'Lobatto'
    else:
        raise NotImplementedError()

    key = str(p[0]) + '-' + str(p[1]) + '-' + btype
    cached, data = ndarray_key_comparer(_cache_bf221i_mq_, [xi_1d, eta_1d], check_str=key)
    if cached:
        return data
    else:
        pass

    if btype == 'Lobatto':
        bfs = (
            Lobatto_polynomials_of_degree(p[0]),
            Lobatto_polynomials_of_degree(p[1]),
        )
    else:
        raise NotImplementedError()

    xi, eta = np.meshgrid(xi_1d, eta_1d, indexing='ij')
    mesh_grid = (xi.ravel('F'), eta.ravel('F'))
    ed_xi = bfs[0].edge_basis(x=xi_1d)
    lb_et = bfs[1].node_basis(x=eta_1d)
    bf_edge_dxi = np.kron(lb_et, ed_xi)
    lb_xi = bfs[0].node_basis(x=xi_1d)
    ed_et = bfs[1].edge_basis(x=eta_1d)
    bf_edge_det = np.kron(ed_et, lb_xi)
    _basis_ = (bf_edge_dxi, bf_edge_det)
    data = mesh_grid, _basis_
    add_to_ndarray_cache(_cache_bf221i_mq_, [xi_1d, eta_1d], data, check_str=key)
    return data
