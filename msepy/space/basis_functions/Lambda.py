# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
from tools.frozen import Frozen
import numpy as np


class MsePyBasisFunctionsLambda(Frozen):
    """"""

    def __init__(self, space, degree):
        """Store required info."""
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._m = space.abstract.m  # esd
        self._orientation = space.abstract.orientation
        self._bfs = space[degree].bfs
        self._freeze()

    def __call__(self, *meshgrid_xi_et_sg):
        """meshgrid means we have to do meshgrid to the 1d xi, et, sg ..."""
        m, n, k = self._m, self._n, self._k

        for i in range(n):
            ref_coo = meshgrid_xi_et_sg[i]
            assert ref_coo.__class__.__name__ in ('list', 'ndarray'), \
                " <Polynomials> : xi_et_sg[{}].type={} is wrong.".format(i, ref_coo.__class__.__name__)
            assert np.ndim(ref_coo) == 1, \
                " <Polynomials> : ndim(xi_et_sg[{}])={} is wrong.".format(i, np.ndim(ref_coo))
            if np.size(ref_coo) > 1:
                assert np.all(np.diff(ref_coo) > 0) and np.max(ref_coo) <= 1 and np.min(ref_coo) >= -1, \
                    " <Polynomials> : ref_coo={} wrong, need to be increasing and bounded in [-1, 1].".format(
                        ref_coo)
            else:
                pass

        if self._n == 2 and self._k == 1:
            return getattr(self, f"_m{m}_n{n}_k{k}_{self._orientation}")(*meshgrid_xi_et_sg)
        else:
            return getattr(self, f"_m{m}_n{n}_k{k}")(*meshgrid_xi_et_sg)

    def _m3_n3_k3(self, *domain):
        """"""
        xi, eta, sigma = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'), sigma.ravel('F'))

        bf_xi = self._bfs[0].edge_basis(x=domain[0])
        bf_et = self._bfs[1].edge_basis(x=domain[1])
        bf_si = self._bfs[2].edge_basis(x=domain[2])
        bf = np.kron(np.kron(bf_si, bf_et), bf_xi)

        _basis_ = (bf,)
        return mesh_grid, _basis_

    def _m3_n3_k2(self, *domain):
        """"""
        xi, eta, sigma = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'), sigma.ravel('F'))

        lb_xi = self._bfs[0].node_basis(x=domain[0])
        ed_et = self._bfs[1].edge_basis(x=domain[1])
        ed_si = self._bfs[2].edge_basis(x=domain[2])
        bf_face_det_dsi = np.kron(np.kron(ed_si, ed_et), lb_xi)

        ed_xi = self._bfs[0].edge_basis(x=domain[0])
        lb_et = self._bfs[1].node_basis(x=domain[1])
        ed_si = self._bfs[2].edge_basis(x=domain[2])
        bf_face_dsi_dxi = np.kron(np.kron(ed_si, lb_et), ed_xi)

        ed_xi = self._bfs[0].edge_basis(x=domain[0])
        ed_et = self._bfs[1].edge_basis(x=domain[1])
        lb_si = self._bfs[2].node_basis(x=domain[2])
        bf_face_dxi_det = np.kron(np.kron(lb_si, ed_et), ed_xi)

        _basis_ = (bf_face_det_dsi, bf_face_dsi_dxi, bf_face_dxi_det)
        return mesh_grid, _basis_

    def _m3_n3_k1(self, *domain):
        """"""
        xi, eta, sigma = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'), sigma.ravel('F'))

        ed_xi = self._bfs[0].edge_basis(x=domain[0])
        lb_et = self._bfs[1].node_basis(x=domain[1])
        lb_si = self._bfs[2].node_basis(x=domain[2])
        bf_edge_dxi = np.kron(np.kron(lb_si, lb_et), ed_xi)

        lb_xi = self._bfs[0].node_basis(x=domain[0])
        ed_et = self._bfs[1].edge_basis(x=domain[1])
        lb_si = self._bfs[2].node_basis(x=domain[2])
        bf_edge_det = np.kron(np.kron(lb_si, ed_et), lb_xi)

        lb_xi = self._bfs[0].node_basis(x=domain[0])
        lb_et = self._bfs[1].node_basis(x=domain[1])
        ed_si = self._bfs[2].edge_basis(x=domain[2])
        bf_edge_dsi = np.kron(np.kron(ed_si, lb_et), lb_xi)

        _basis_ = (bf_edge_dxi, bf_edge_det, bf_edge_dsi)
        return mesh_grid, _basis_

    def _m3_n3_k0(self, *domain):
        """"""
        xi, eta, sigma = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'), sigma.ravel('F'))

        bf_xi = self._bfs[0].node_basis(x=domain[0])
        bf_et = self._bfs[1].node_basis(x=domain[1])
        bf_si = self._bfs[2].node_basis(x=domain[2])
        bf = np.kron(np.kron(bf_si, bf_et), bf_xi)

        _basis_ = (bf,)
        return mesh_grid, _basis_

    def _m2_n2_k2(self, *domain):
        """"""
        xi, eta = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'))
        bf_xi = self._bfs[0].edge_basis(x=domain[0])
        bf_et = self._bfs[1].edge_basis(x=domain[1])
        bf = np.kron(bf_et, bf_xi)
        _basis_ = (bf,)
        return mesh_grid, _basis_

    def _m2_n2_k1_inner(self, *domain):
        """"""
        xi, eta = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'))
        ed_xi = self._bfs[0].edge_basis(x=domain[0])
        lb_et = self._bfs[1].node_basis(x=domain[1])
        bf_edge_dxi = np.kron(lb_et, ed_xi)
        lb_xi = self._bfs[0].node_basis(x=domain[0])
        ed_et = self._bfs[1].edge_basis(x=domain[1])
        bf_edge_det = np.kron(ed_et, lb_xi)
        _basis_ = (bf_edge_dxi, bf_edge_det)
        return mesh_grid, _basis_

    def _m2_n2_k1_outer(self, *domain):
        """"""
        xi, eta = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'))
        lb_xi = self._bfs[0].node_basis(x=domain[0])
        ed_et = self._bfs[1].edge_basis(x=domain[1])
        bf_edge_det = np.kron(ed_et, lb_xi)
        ed_xi = self._bfs[0].edge_basis(x=domain[0])
        lb_et = self._bfs[1].node_basis(x=domain[1])
        bf_edge_dxi = np.kron(lb_et, ed_xi)
        _basis_ = (bf_edge_det, bf_edge_dxi)
        return mesh_grid, _basis_

    def _m2_n2_k0(self, *domain):
        """"""
        xi, eta = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'))
        bf_xi = self._bfs[0].node_basis(x=domain[0])
        bf_et = self._bfs[1].node_basis(x=domain[1])
        bf = np.kron(bf_et, bf_xi)
        _basis_ = (bf,)
        return mesh_grid, _basis_

    def _m1_n1_k1(self, *domain):
        """"""
        _basis_ = self._bfs[0].edge_basis(domain[0])
        return domain, (_basis_,)

    def _m1_n1_k0(self, *domain):
        """"""
        _basis_ = self._bfs[0].node_basis(domain[0])
        return domain, (_basis_,)
