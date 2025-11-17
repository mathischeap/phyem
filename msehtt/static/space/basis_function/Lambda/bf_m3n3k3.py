# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.msepy.tools.polynomials import Lobatto_polynomials_of_degree
from phyem.tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer


_cache_bf333_mq_ = {}


def ___bf333_msepy_quadrilateral___(p, btype, xi_1d, eta_1d, sg_1d):
    """"""
    key = str(p[0]) + '-' + str(p[1]) + '-' + str(p[2]) + '-' + btype

    cached, data = ndarray_key_comparer(_cache_bf333_mq_, [xi_1d, eta_1d, sg_1d], check_str=key)
    if cached:
        return data
    else:
        pass

    if btype == 'Lobatto':
        bfs = (
            Lobatto_polynomials_of_degree(p[0]),
            Lobatto_polynomials_of_degree(p[1]),
            Lobatto_polynomials_of_degree(p[2]),
        )
    else:
        raise NotImplementedError()

    xi, eta, sg = np.meshgrid(xi_1d, eta_1d, sg_1d, indexing='ij')
    mesh_grid = (xi.ravel('F'), eta.ravel('F'), sg.ravel('F'))

    bf_xi = bfs[0].edge_basis(x=xi_1d)
    bf_et = bfs[1].edge_basis(x=eta_1d)
    bf_sg = bfs[2].edge_basis(x=sg_1d)

    bf = np.kron(np.kron(bf_sg, bf_et), bf_xi)
    _basis_ = (bf,)
    data = mesh_grid, _basis_
    add_to_ndarray_cache(_cache_bf333_mq_, [xi_1d, eta_1d, sg_1d], data, check_str=key)
    return data
