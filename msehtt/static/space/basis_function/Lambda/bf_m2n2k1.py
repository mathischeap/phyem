# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.msepy.tools.polynomials import Lobatto_polynomials_of_degree, polynomials_on_nodes
from phyem.tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer


# ------- OUTER ---------------------------------------------------------------------------------


_cache_bf221o_mq_ = {}


def ___bf221o_outer_msepy_quadrilateral___(p, btype, xi_1d, eta_1d):
    """"""
    key = str(p[0]) + '-' + str(p[1]) + '-' + str(btype)
    cached, data = ndarray_key_comparer(_cache_bf221o_mq_, [xi_1d, eta_1d], check_str=key)
    if cached:
        return data
    else:
        pass

    if btype == ('Lobatto', 'Lobatto'):
        bfs = (
            Lobatto_polynomials_of_degree(p[0]),
            Lobatto_polynomials_of_degree(p[1]),
        )
    else:
        bfs = []
        for pi, bt in enumerate(btype):
            if bt == 'Lobatto':
                BasisFunction = Lobatto_polynomials_of_degree(pi)
            else:
                BasisFunction = polynomials_on_nodes(bt)
            bfs.append(BasisFunction)
        bfs = tuple(bfs)

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


_cache_bf221o_vtu5_ = {}


def ___bf221o_outer_vtu_5___(p, btype, xi_1d, eta_1d):
    """"""
    key = str(p[0]) + '-' + str(p[1]) + '-' + str(btype)
    cached, data = ndarray_key_comparer(_cache_bf221o_vtu5_, [xi_1d, eta_1d], check_str=key)
    if cached:
        return data
    else:
        pass

    if btype == ('Lobatto', 'Lobatto'):
        bfs = (
            Lobatto_polynomials_of_degree(p[0]),
            Lobatto_polynomials_of_degree(p[1]),
        )
    else:
        bfs = []
        for pi, bt in enumerate(btype):
            if bt == 'Lobatto':
                BasisFunction = Lobatto_polynomials_of_degree(pi)
            else:
                BasisFunction = polynomials_on_nodes(bt)
            bfs.append(BasisFunction)
        bfs = tuple(bfs)

    xi, eta = np.meshgrid(xi_1d, eta_1d, indexing='ij')
    mesh_grid = (xi.ravel('F'), eta.ravel('F'))

    lb_xi = bfs[0].node_basis(x=xi_1d)[1:, :]   # collapsed here!
    ed_et = bfs[1].edge_basis(x=eta_1d)
    bf_edge_det = np.kron(ed_et, lb_xi)

    ed_xi = bfs[0].edge_basis(x=xi_1d)
    lb_et = bfs[1].node_basis(x=eta_1d)
    bf_edge_dxi = np.kron(lb_et, ed_xi)

    _basis_ = (bf_edge_det, bf_edge_dxi)
    data = mesh_grid, _basis_
    add_to_ndarray_cache(_cache_bf221o_vtu5_, [xi_1d, eta_1d], data, check_str=key)
    return data


# ------- INNER ---------------------------------------------------------------------------------


_cache_bf221i_mq_ = {}


def ___bf221i_inner_msepy_quadrilateral___(p, btype, xi_1d, eta_1d):
    r""""""
    key = str(p[0]) + '-' + str(p[1]) + '-' + str(btype)
    cached, data = ndarray_key_comparer(_cache_bf221i_mq_, [xi_1d, eta_1d], check_str=key)
    if cached:
        return data
    else:
        pass

    if btype == ('Lobatto', 'Lobatto'):
        bfs = (
            Lobatto_polynomials_of_degree(p[0]),
            Lobatto_polynomials_of_degree(p[1]),
        )
    else:
        bfs = []
        for pi, bt in enumerate(btype):
            if bt == 'Lobatto':
                BasisFunction = Lobatto_polynomials_of_degree(pi)
            else:
                BasisFunction = polynomials_on_nodes(bt)
            bfs.append(BasisFunction)
        bfs = tuple(bfs)

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


_cache_bf221i_vtu5_ = {}


def ___bf221i_inner_vtu_5___(p, btype, xi_1d, eta_1d):
    r""""""
    key = str(p[0]) + '-' + str(p[1]) + '-' + str(btype)
    cached, data = ndarray_key_comparer(_cache_bf221i_vtu5_, [xi_1d, eta_1d], check_str=key)
    if cached:
        return data
    else:
        pass

    if btype == ('Lobatto', 'Lobatto'):
        bfs = (
            Lobatto_polynomials_of_degree(p[0]),
            Lobatto_polynomials_of_degree(p[1]),
        )
    else:
        bfs = []
        for pi, bt in enumerate(btype):
            if bt == 'Lobatto':
                BasisFunction = Lobatto_polynomials_of_degree(pi)
            else:
                BasisFunction = polynomials_on_nodes(bt)
            bfs.append(BasisFunction)
        bfs = tuple(bfs)

    xi, eta = np.meshgrid(xi_1d, eta_1d, indexing='ij')
    mesh_grid = (xi.ravel('F'), eta.ravel('F'))

    ed_xi = bfs[0].edge_basis(x=xi_1d)
    lb_et = bfs[1].node_basis(x=eta_1d)
    bf_edge_dxi = np.kron(lb_et, ed_xi)

    lb_xi = bfs[0].node_basis(x=xi_1d)[1:, :]   # collapsed here!
    ed_et = bfs[1].edge_basis(x=eta_1d)
    bf_edge_det = np.kron(ed_et, lb_xi)

    _basis_ = (bf_edge_dxi, bf_edge_det)
    data = mesh_grid, _basis_
    add_to_ndarray_cache(_cache_bf221i_vtu5_, [xi_1d, eta_1d], data, check_str=key)
    return data
