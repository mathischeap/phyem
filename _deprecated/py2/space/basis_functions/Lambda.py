# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import numpy as np
from tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer


class MseHyPy2BasisFunctionsLambda(Frozen):
    """"""

    def __init__(self, space, degree):
        r"""Store required info."""
        self._k = space.abstract.k
        self._orientation = space.abstract.orientation
        self._bfs = space[degree].bfs
        self._space = space
        self._bfs_cache = dict()
        self._freeze()

    def __call__(self, g, xi, et):
        r"""meshgrid means we have to do meshgrid to the 1d xi, et, sg ..."""
        _ = self._space.mesh._pg(g)
        # just to check g is correct or not, also to remind user that basis function will be mesh-dependent.

        cached, bfs = ndarray_key_comparer(self._bfs_cache, [xi, et])
        if cached:
            pass
        else:

            k = self._k

            for i in range(2):
                ref_coo = (xi, et)[i]
                assert ref_coo.__class__.__name__ in ('list', 'ndarray'), \
                    " <bf> : xi_et_sg[{}].type={} is wrong.".format(
                        i, ref_coo.__class__.__name__
                    )
                assert np.ndim(ref_coo) == 1, \
                    " <bf> : ndim(xi_et_sg[{}])={} is wrong.".format(
                        i, np.ndim(ref_coo)
                    )
                if np.size(ref_coo) > 1:
                    assert np.all(np.diff(ref_coo) > 0) and np.max(ref_coo) <= 1 and np.min(ref_coo) >= -1, \
                        " <bf> : ref_coo={} wrong, need to be increasing and bounded in [-1, 1].".format(
                            ref_coo)
                else:
                    pass

            if self._k == 1:
                bfs = getattr(self, f"_k{k}_{self._orientation}")(xi, et)
            else:
                bfs = getattr(self, f"_k{k}")(xi, et)

            add_to_ndarray_cache(self._bfs_cache, [xi, et], bfs)

        return bfs

    def _k2(self, *domain):
        r""""""
        xi, eta = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'))

        bf_xi = self._bfs[0].edge_basis(x=domain[0])
        bf_et = self._bfs[1].edge_basis(x=domain[1])
        bf = np.kron(bf_et, bf_xi)
        _basis_ = (bf,)

        bf = {
            'q': _basis_,   # same for quadrilateral or triangle cell
            't': _basis_,   # same for quadrilateral or triangle cell
        }
        return mesh_grid, bf

    def _k1_inner(self, *domain):
        r"""For example, p = 3

        component 0; dx:

        ^ y
        |
        |
                     11
                 10
               9
               6  7  8
        top <  3  4  5
               0
                  1
                     2
        -------------------------> x

        component 1; dy:

        ^ y
        |
        |
                    8
                  7
               6
        top <  3  4 5
               0
                  1
                    2
        -------------------------> x

        """
        xi, eta = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'))

        ed_xi = self._bfs[0].edge_basis(x=domain[0])
        lb_et = self._bfs[1].node_basis(x=domain[1])
        lb_xi = self._bfs[0].node_basis(x=domain[0])
        ed_et = self._bfs[1].edge_basis(x=domain[1])

        bf_edge_dxi = np.kron(lb_et, ed_xi)
        bf_edge_det = np.kron(ed_et, lb_xi)
        _basis_q_ = (bf_edge_dxi, bf_edge_det)

        basis_dx = np.kron(lb_et, ed_xi)
        basis_dy = np.kron(ed_et, lb_xi[1:, :])
        _basis_t_ = (basis_dx, basis_dy)

        bf = {
            'q': _basis_q_,   # for quadrilateral or triangle cell
            't': _basis_t_,   # for quadrilateral or triangle cell
        }
        return mesh_grid, bf

    def _k1_outer(self, *domain):
        r"""For example, p = 3

        component 0; dy:

        ^ y
        |
        |
                    8
                  7
               6
        top <  3  4 5
               0
                  1
                    2
        -------------------------> x

        component 1; dx:

        ^ y
        |
        |
                      11
                  10
               9
               6  7   8
        top <
               3  4  5
               0
                  1
                     2
        -------------------------> x

        Parameters
        ----------
        domain

        Returns
        -------

        """
        xi, eta = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'))

        lb_xi = self._bfs[0].node_basis(x=domain[0])
        ed_et = self._bfs[1].edge_basis(x=domain[1])

        ed_xi = self._bfs[0].edge_basis(x=domain[0])
        lb_et = self._bfs[1].node_basis(x=domain[1])

        bf_edge_det = np.kron(ed_et, lb_xi)
        bf_edge_dxi = np.kron(lb_et, ed_xi)
        _basis_q_ = (bf_edge_det, bf_edge_dxi)

        basis_dy = np.kron(ed_et, lb_xi[1:, :])
        basis_dx = np.kron(lb_et, ed_xi)
        _basis_t_ = (basis_dy, basis_dx)

        bf = {
            'q': _basis_q_,  # for quadrilateral or triangle cell
            't': _basis_t_,  # for quadrilateral or triangle cell
        }
        return mesh_grid, bf

    def _k0(self, *domain):
        r"""
        ^ y
        |
        |
            /|
           / |
          /. |
        0/  .|
         \ 4 |.
         1\ 5|
           \ |6
          2 \|
             3

        o---------------> x

        Parameters
        ----------
        domain

        Returns
        -------

        """
        xi, eta = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'))

        bf_xi = self._bfs[0].node_basis(x=domain[0])
        bf_et = self._bfs[1].node_basis(x=domain[1])

        bf = np.kron(bf_et, bf_xi)
        _basis_q_ = (bf,)

        basis_singular = np.sum(np.kron(bf_et, bf_xi[0, :]), axis=0)[np.newaxis, :]
        basis_regular = np.kron(bf_et, bf_xi[1:, :])
        _basis_t_ = (np.vstack((basis_singular, basis_regular)),)

        bf = {
            'q': _basis_q_,   # for quadrilateral or triangle cell
            't': _basis_t_,   # for quadrilateral or triangle cell
        }

        return mesh_grid, bf
