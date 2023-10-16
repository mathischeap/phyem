# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import numpy as np


class BasisFunctionsLambda(Frozen):
    """"""

    def __init__(self, space):
        r"""Store required info."""
        self._k = space.abstract.k
        self._orientation = space.abstract.orientation
        self._space = space
        self._bfs = None
        self._freeze()

    def __call__(self, degree, *meshgrid_xi_et):
        r"""meshgrid means we have to do meshgrid to the 1d xi, et, sg ..."""
        self._bfs = self._space[degree].bfs

        for i in range(2):
            ref_coo = meshgrid_xi_et[i]
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

        print(111)
        if self._k == 1:
            return getattr(self, f"_k{self._k}_{self._orientation}")(*meshgrid_xi_et)
        else:
            return getattr(self, f"_k{self._k}")(*meshgrid_xi_et)

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
