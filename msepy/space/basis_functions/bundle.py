# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen


class MsePyBasisFunctionsBundle(Frozen):
    """"""

    def __init__(self, space, degree):
        r"""Store required info."""
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._m = space.abstract.m  # esd
        self._orientation = space.abstract.orientation
        self._bfs = space[degree].bfs
        self._freeze()

    def __call__(self, *meshgrid_xi_et_sg):
        r"""meshgrid means we have to do meshgrid to the 1d xi, et, sg ..."""
        m, n, k = self._m, self._n, self._k

        for i in range(n):  # do some regular checks.
            ref_coo = meshgrid_xi_et_sg[i]
            assert ref_coo.__class__.__name__ in ('list', 'ndarray'), \
                " <Polynomials> : xi_et_sg[{}].type={} is wrong.".format(
                    i, ref_coo.__class__.__name__
                )
            assert np.ndim(ref_coo) == 1, \
                " <Polynomials> : ndim(xi_et_sg[{}])={} is wrong.".format(
                    i, np.ndim(ref_coo)
                )
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
        r""""""
        xi, eta, sigma = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'), sigma.ravel('F'))

        _bfs = self._bfs[0]
        bf_xi = _bfs[0].edge_basis(x=domain[0])
        bf_et = _bfs[1].edge_basis(x=domain[1])
        bf_si = _bfs[2].edge_basis(x=domain[2])
        bf0 = np.kron(np.kron(bf_si, bf_et), bf_xi)

        _bfs = self._bfs[1]
        bf_xi = _bfs[0].edge_basis(x=domain[0])
        bf_et = _bfs[1].edge_basis(x=domain[1])
        bf_si = _bfs[2].edge_basis(x=domain[2])
        bf1 = np.kron(np.kron(bf_si, bf_et), bf_xi)

        _bfs = self._bfs[2]
        bf_xi = _bfs[0].edge_basis(x=domain[0])
        bf_et = _bfs[1].edge_basis(x=domain[1])
        bf_si = _bfs[2].edge_basis(x=domain[2])
        bf2 = np.kron(np.kron(bf_si, bf_et), bf_xi)

        _basis_ = (bf0, bf1, bf2)
        return mesh_grid, _basis_

    def _m3_n3_k2(self, *domain):
        r""""""
        xi, eta, sigma = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'), sigma.ravel('F'))

        _bfs = self._bfs[0]
        lb_xi = _bfs[0].node_basis(x=domain[0])
        ed_et = _bfs[1].edge_basis(x=domain[1])
        ed_si = _bfs[2].edge_basis(x=domain[2])
        bf_face_det_dsi = np.kron(np.kron(ed_si, ed_et), lb_xi)
        ed_xi = _bfs[0].edge_basis(x=domain[0])
        lb_et = _bfs[1].node_basis(x=domain[1])
        ed_si = _bfs[2].edge_basis(x=domain[2])
        bf_face_dsi_dxi = np.kron(np.kron(ed_si, lb_et), ed_xi)
        ed_xi = _bfs[0].edge_basis(x=domain[0])
        ed_et = _bfs[1].edge_basis(x=domain[1])
        lb_si = _bfs[2].node_basis(x=domain[2])
        bf_face_dxi_det = np.kron(np.kron(lb_si, ed_et), ed_xi)
        _basis0_ = (bf_face_det_dsi, bf_face_dsi_dxi, bf_face_dxi_det)

        _bfs = self._bfs[1]
        lb_xi = _bfs[0].node_basis(x=domain[0])
        ed_et = _bfs[1].edge_basis(x=domain[1])
        ed_si = _bfs[2].edge_basis(x=domain[2])
        bf_face_det_dsi = np.kron(np.kron(ed_si, ed_et), lb_xi)
        ed_xi = _bfs[0].edge_basis(x=domain[0])
        lb_et = _bfs[1].node_basis(x=domain[1])
        ed_si = _bfs[2].edge_basis(x=domain[2])
        bf_face_dsi_dxi = np.kron(np.kron(ed_si, lb_et), ed_xi)
        ed_xi = _bfs[0].edge_basis(x=domain[0])
        ed_et = _bfs[1].edge_basis(x=domain[1])
        lb_si = _bfs[2].node_basis(x=domain[2])
        bf_face_dxi_det = np.kron(np.kron(lb_si, ed_et), ed_xi)
        _basis1_ = (bf_face_det_dsi, bf_face_dsi_dxi, bf_face_dxi_det)

        _bfs = self._bfs[2]
        lb_xi = _bfs[0].node_basis(x=domain[0])
        ed_et = _bfs[1].edge_basis(x=domain[1])
        ed_si = _bfs[2].edge_basis(x=domain[2])
        bf_face_det_dsi = np.kron(np.kron(ed_si, ed_et), lb_xi)
        ed_xi = _bfs[0].edge_basis(x=domain[0])
        lb_et = _bfs[1].node_basis(x=domain[1])
        ed_si = _bfs[2].edge_basis(x=domain[2])
        bf_face_dsi_dxi = np.kron(np.kron(ed_si, lb_et), ed_xi)
        ed_xi = _bfs[0].edge_basis(x=domain[0])
        ed_et = _bfs[1].edge_basis(x=domain[1])
        lb_si = _bfs[2].node_basis(x=domain[2])
        bf_face_dxi_det = np.kron(np.kron(lb_si, ed_et), ed_xi)
        _basis2_ = (bf_face_det_dsi, bf_face_dsi_dxi, bf_face_dxi_det)

        _basis_ = (_basis0_, _basis1_, _basis2_)
        return mesh_grid, _basis_

    def _m3_n3_k1(self, *domain):
        r""""""
        xi, eta, sigma = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'), sigma.ravel('F'))

        _bfs = self._bfs[0]
        ed_xi = _bfs[0].edge_basis(x=domain[0])
        lb_et = _bfs[1].node_basis(x=domain[1])
        lb_si = _bfs[2].node_basis(x=domain[2])
        bf_edge_dxi = np.kron(np.kron(lb_si, lb_et), ed_xi)
        lb_xi = _bfs[0].node_basis(x=domain[0])
        ed_et = _bfs[1].edge_basis(x=domain[1])
        lb_si = _bfs[2].node_basis(x=domain[2])
        bf_edge_det = np.kron(np.kron(lb_si, ed_et), lb_xi)
        lb_xi = _bfs[0].node_basis(x=domain[0])
        lb_et = _bfs[1].node_basis(x=domain[1])
        ed_si = _bfs[2].edge_basis(x=domain[2])
        bf_edge_dsi = np.kron(np.kron(ed_si, lb_et), lb_xi)
        _basis0_ = (bf_edge_dxi, bf_edge_det, bf_edge_dsi)

        _bfs = self._bfs[1]
        ed_xi = _bfs[0].edge_basis(x=domain[0])
        lb_et = _bfs[1].node_basis(x=domain[1])
        lb_si = _bfs[2].node_basis(x=domain[2])
        bf_edge_dxi = np.kron(np.kron(lb_si, lb_et), ed_xi)
        lb_xi = _bfs[0].node_basis(x=domain[0])
        ed_et = _bfs[1].edge_basis(x=domain[1])
        lb_si = _bfs[2].node_basis(x=domain[2])
        bf_edge_det = np.kron(np.kron(lb_si, ed_et), lb_xi)
        lb_xi = _bfs[0].node_basis(x=domain[0])
        lb_et = _bfs[1].node_basis(x=domain[1])
        ed_si = _bfs[2].edge_basis(x=domain[2])
        bf_edge_dsi = np.kron(np.kron(ed_si, lb_et), lb_xi)
        _basis1_ = (bf_edge_dxi, bf_edge_det, bf_edge_dsi)

        _bfs = self._bfs[2]
        ed_xi = _bfs[0].edge_basis(x=domain[0])
        lb_et = _bfs[1].node_basis(x=domain[1])
        lb_si = _bfs[2].node_basis(x=domain[2])
        bf_edge_dxi = np.kron(np.kron(lb_si, lb_et), ed_xi)
        lb_xi = _bfs[0].node_basis(x=domain[0])
        ed_et = _bfs[1].edge_basis(x=domain[1])
        lb_si = _bfs[2].node_basis(x=domain[2])
        bf_edge_det = np.kron(np.kron(lb_si, ed_et), lb_xi)
        lb_xi = _bfs[0].node_basis(x=domain[0])
        lb_et = _bfs[1].node_basis(x=domain[1])
        ed_si = _bfs[2].edge_basis(x=domain[2])
        bf_edge_dsi = np.kron(np.kron(ed_si, lb_et), lb_xi)
        _basis2_ = (bf_edge_dxi, bf_edge_det, bf_edge_dsi)

        _basis_ = (_basis0_, _basis1_, _basis2_)
        return mesh_grid, _basis_

    def _m3_n3_k0(self, *domain):
        r""""""
        xi, eta, sigma = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'), sigma.ravel('F'))

        _bfs = self._bfs[0]
        bf_xi = _bfs[0].node_basis(x=domain[0])
        bf_et = _bfs[1].node_basis(x=domain[1])
        bf_si = _bfs[2].node_basis(x=domain[2])
        bf0 = np.kron(np.kron(bf_si, bf_et), bf_xi)

        _bfs = self._bfs[1]
        bf_xi = _bfs[0].node_basis(x=domain[0])
        bf_et = _bfs[1].node_basis(x=domain[1])
        bf_si = _bfs[2].node_basis(x=domain[2])
        bf1 = np.kron(np.kron(bf_si, bf_et), bf_xi)

        _bfs = self._bfs[2]
        bf_xi = _bfs[0].node_basis(x=domain[0])
        bf_et = _bfs[1].node_basis(x=domain[1])
        bf_si = _bfs[2].node_basis(x=domain[2])
        bf2 = np.kron(np.kron(bf_si, bf_et), bf_xi)

        _basis_ = (bf0, bf1, bf2)
        return mesh_grid, _basis_

    def _m2_n2_k2(self, *domain):
        r""""""
        xi, eta = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'))

        _bfs = self._bfs[0]
        bf_xi = _bfs[0].edge_basis(x=domain[0])
        bf_et = _bfs[1].edge_basis(x=domain[1])
        bf0 = np.kron(bf_et, bf_xi)

        _bfs = self._bfs[1]
        bf_xi = _bfs[0].edge_basis(x=domain[0])
        bf_et = _bfs[1].edge_basis(x=domain[1])
        bf1 = np.kron(bf_et, bf_xi)

        _basis_ = (bf0, bf1)
        return mesh_grid, _basis_

    def _m2_n2_k1_inner(self, *domain):
        r""""""
        xi, eta = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'))

        _bfs = self._bfs[0]
        ed_xi = _bfs[0].edge_basis(x=domain[0])
        lb_et = _bfs[1].node_basis(x=domain[1])
        bf_edge_dxi = np.kron(lb_et, ed_xi)
        lb_xi = _bfs[0].node_basis(x=domain[0])
        ed_et = _bfs[1].edge_basis(x=domain[1])
        bf_edge_det = np.kron(ed_et, lb_xi)
        _basis0_ = (bf_edge_dxi, bf_edge_det)

        _bfs = self._bfs[1]
        ed_xi = _bfs[0].edge_basis(x=domain[0])
        lb_et = _bfs[1].node_basis(x=domain[1])
        bf_edge_dxi = np.kron(lb_et, ed_xi)
        lb_xi = _bfs[0].node_basis(x=domain[0])
        ed_et = _bfs[1].edge_basis(x=domain[1])
        bf_edge_det = np.kron(ed_et, lb_xi)
        _basis1_ = (bf_edge_dxi, bf_edge_det)

        return mesh_grid, (_basis0_, _basis1_)

    def _m2_n2_k1_outer(self, *domain):
        r""""""
        xi, eta = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'))

        _bfs = self._bfs[0]
        lb_xi = _bfs[0].node_basis(x=domain[0])
        ed_et = _bfs[1].edge_basis(x=domain[1])
        bf_edge_det = np.kron(ed_et, lb_xi)
        ed_xi = _bfs[0].edge_basis(x=domain[0])
        lb_et = _bfs[1].node_basis(x=domain[1])
        bf_edge_dxi = np.kron(lb_et, ed_xi)
        _basis0_ = (bf_edge_det, bf_edge_dxi)

        _bfs = self._bfs[1]
        lb_xi = _bfs[0].node_basis(x=domain[0])
        ed_et = _bfs[1].edge_basis(x=domain[1])
        bf_edge_det = np.kron(ed_et, lb_xi)
        ed_xi = _bfs[0].edge_basis(x=domain[0])
        lb_et = _bfs[1].node_basis(x=domain[1])
        bf_edge_dxi = np.kron(lb_et, ed_xi)
        _basis1_ = (bf_edge_det, bf_edge_dxi)

        return mesh_grid, (_basis0_, _basis1_)

    def _m2_n2_k0(self, *domain):
        r""""""
        xi, eta = np.meshgrid(*domain, indexing='ij')
        mesh_grid = (xi.ravel('F'), eta.ravel('F'))

        _bfs = self._bfs[0]
        bf_xi = _bfs[0].node_basis(x=domain[0])
        bf_et = _bfs[1].node_basis(x=domain[1])
        bf0 = np.kron(bf_et, bf_xi)

        _bfs = self._bfs[1]
        bf_xi = _bfs[0].node_basis(x=domain[0])
        bf_et = _bfs[1].node_basis(x=domain[1])
        bf1 = np.kron(bf_et, bf_xi)

        _basis_ = (bf0, bf1)
        return mesh_grid, _basis_

    def _m1_n1_k1(self, *domain):
        r""""""
        _basis_ = self._bfs[0][0].edge_basis(domain[0])
        return domain, (_basis_,)

    def _m1_n1_k0(self, *domain):
        r""""""
        _basis_ = self._bfs[0][0].node_basis(domain[0])
        return domain, (_basis_,)
