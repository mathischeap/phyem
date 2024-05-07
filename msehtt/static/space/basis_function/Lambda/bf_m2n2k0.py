# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from msepy.tools.polynomials import Lobatto_polynomials_of_degree
from tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer


_cache_bf220_mq_ = {}


def ___bf220_msepy_quadrilateral___(p, btype, xi_1d, eta_1d):
    """"""

    key = str(p[0]) + '-' + str(p[1]) + '-' + btype
    cached, data = ndarray_key_comparer(_cache_bf220_mq_, [xi_1d, eta_1d], check_str=key)
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
    bf_xi = bfs[0].node_basis(x=xi_1d)
    bf_et = bfs[1].node_basis(x=eta_1d)
    bf = np.kron(bf_et, bf_xi)
    _basis_ = (bf,)
    data = mesh_grid, _basis_
    add_to_ndarray_cache(_cache_bf220_mq_, [xi_1d, eta_1d], data, check_str=key)
    return data
