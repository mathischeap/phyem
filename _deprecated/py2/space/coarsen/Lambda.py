# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from tools.quadrature import Quadrature


class MseHyPy2CoarsenLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        indicator = space.abstract.indicator
        assert indicator == 'Lambda', f"must be"
        self._k = space.abstract.k
        self._orientation = space.abstract.orientation
        self._mesh = space.mesh
        self._cache_p0_ = dict()
        self._cache_p2_ = dict()
        self._freeze()

    def __call__(self, degree, dest_g__dest_index, source_g__source_indices__cochain):
        """"""
        dest_g, dest_index = dest_g__dest_index
        source_g, source_indices, source_cochain = source_g__source_indices__cochain
        assert isinstance(source_cochain, dict) and all([_ in source_cochain for _ in source_indices]), \
            f"source cochain do not match source indices"
        assert len(source_cochain) == len(source_indices), f"source cochain do not match source indices"
        assert dest_g != source_g, f"source g must be different from dest g."

        mesh = self._mesh
        dest_index = mesh[dest_g][dest_index].index

        if self._k == 0:
            return self._k0(
                degree, dest_index, source_indices, source_cochain, (dest_g, source_g))
        elif self._k == 1 and self._orientation == 'inner':
            return self._k1_inner(
                degree, dest_index, source_indices, source_cochain, (dest_g, source_g))
        elif self._k == 1 and self._orientation == 'outer':
            return self._k1_outer(
                degree, dest_index, source_indices, source_cochain, (dest_g, source_g))
        elif self._k == 2:
            return self._k2(
                degree, dest_index, source_indices, source_cochain, (dest_g, source_g))
        else:
            raise Exception

    def _k1_inner(self, degree, dest_index, source_indices, source_cochain, dg_sg, method='interpolation'):
        """coarsen scalar valued 0-forms to `d_fc` from `s_fcs` with cochain `source_cochain`"""
        if method == 'interpolation':
            cochain = self._k1_inner_interpolation(degree, dest_index, source_indices, source_cochain, dg_sg)
        else:
            raise NotImplementedError(f"method={method} not implemented for 0-form coarsen")

        return cochain

    def _k1_inner_interpolation(self, degree, dest_index, source_indices, source_cochain, dg_sg):
        """"""
        xi_and_et = self._parse_p_for_mesh_grid_1_2(degree)
        dg, sg = dg_sg
        xy, v = self._space.reconstruct(
            sg, source_cochain, xi_and_et, xi_and_et, ravel=True,
            fc_range=source_indices, degree=degree
        )
        x, y = xy
        v0, v1 = v
        x_list = list()
        y_list = list()
        v0_list = list()
        v1_list = list()
        for index in source_indices:
            x_list.extend(x[index])
            y_list.extend(y[index])
            v0_list.extend(v0[index])
            v1_list.extend(v1[index])
        interp0 = NearestNDInterpolator(list(zip(x_list, y_list)), v0_list)
        interp1 = NearestNDInterpolator(list(zip(x_list, y_list)), v1_list)
        _ = self._space.reduce.Lambda._local_k1_inner_evolve([interp0, interp1], dg, degree, fc_range=[dest_index])
        cochain = _[dest_index]
        return cochain

    def _k1_outer(self, degree, dest_index, source_indices, source_cochain, dg_sg, method='interpolation'):
        """coarsen scalar valued 0-forms to `d_fc` from `s_fcs` with cochain `source_cochain`"""
        if method == 'interpolation':
            cochain = self._k1_outer_interpolation(degree, dest_index, source_indices, source_cochain, dg_sg)
        else:
            raise NotImplementedError(f"method={method} not implemented for 0-form coarsen")

        return cochain

    def _k1_outer_interpolation(self, degree, dest_index, source_indices, source_cochain, dg_sg):
        """"""
        xi_and_et = self._parse_p_for_mesh_grid_1_2(degree)
        dg, sg = dg_sg
        xy, v = self._space.reconstruct(
            sg, source_cochain, xi_and_et, xi_and_et, ravel=True,
            fc_range=source_indices, degree=degree
        )
        x, y = xy
        v0, v1 = v
        x_list = list()
        y_list = list()
        v0_list = list()
        v1_list = list()
        for index in source_indices:
            x_list.extend(x[index])
            y_list.extend(y[index])
            v0_list.extend(v0[index])
            v1_list.extend(v1[index])
        interp0 = NearestNDInterpolator(list(zip(x_list, y_list)), v0_list)
        interp1 = NearestNDInterpolator(list(zip(x_list, y_list)), v1_list)
        _ = self._space.reduce.Lambda._local_k1_outer_evolve([interp0, interp1], dg, degree, fc_range=[dest_index])
        cochain = _[dest_index]
        return cochain

    def _k2(self, degree, dest_index, source_indices, source_cochain, dg_sg, method='interpolation'):
        """coarsen scalar valued 0-forms to `d_fc` from `s_fcs` with cochain `source_cochain`"""
        if method == 'interpolation':
            cochain = self._k2_interpolation(degree, dest_index, source_indices, source_cochain, dg_sg)
        else:
            raise NotImplementedError(f"method={method} not implemented for 0-form coarsen")

        return cochain

    def _parse_p_for_mesh_grid_1_2(self, degree):
        """"""
        p = self._space[degree].p
        if p in self._cache_p2_:
            pass
        else:
            px, py = p
            assert px == py
            self._cache_p2_[p] = Quadrature(2 * px + 7, category='Gauss').quad_nodes
        return self._cache_p2_[p]

    def _k2_interpolation(self, degree, dest_index, source_indices, source_cochain, dg_sg):
        """"""
        xi_and_et = self._parse_p_for_mesh_grid_1_2(degree)
        dg, sg = dg_sg
        xy, v = self._space.reconstruct(
            sg, source_cochain, xi_and_et, xi_and_et, ravel=True,
            fc_range=source_indices, degree=degree
        )
        x, y = xy
        v = v[0]
        x_list = list()
        y_list = list()
        v_list = list()
        for index in source_indices:
            x_list.extend(x[index])
            y_list.extend(y[index])
            v_list.extend(v[index])
        interp = NearestNDInterpolator(list(zip(x_list, y_list)), v_list)
        _ = self._space.reduce.Lambda._local_k2(interp, dg, degree, fc_range=[dest_index])
        cochain = _[dest_index]
        return cochain

    def _k0(self, degree, dest_index, source_indices, source_cochain, dg_sg, method='interpolation'):
        """coarsen scalar valued 0-forms to `d_fc` from `s_fcs` with cochain `source_cochain`"""
        if method == 'interpolation':
            cochain = self._k0_interpolation(degree, dest_index, source_indices, source_cochain, dg_sg)
        else:
            raise NotImplementedError(f"method={method} not implemented for 0-form coarsen")

        return cochain

    def _parse_p_for_mesh_grid_0(self, degree):
        """"""
        p = self._space[degree].p
        if p in self._cache_p0_:
            pass
        else:
            px, py = p
            assert px == py
            self._cache_p0_[p] = np.linspace(-1, 1, 5*px + 15)
        return self._cache_p0_[p]

    def _k0_interpolation(self, degree, dest_index, source_indices, source_cochain, dg_sg):
        """"""
        xi_and_et = self._parse_p_for_mesh_grid_0(degree)
        dg, sg = dg_sg
        xy, v = self._space.reconstruct(
            sg, source_cochain, xi_and_et, xi_and_et, ravel=True,
            fc_range=source_indices, degree=degree
        )
        x, y = xy
        v = v[0]
        x_list = list()
        y_list = list()
        v_list = list()
        for index in source_indices:
            x_list.extend(x[index])
            y_list.extend(y[index])
            v_list.extend(v[index])
        interp = NearestNDInterpolator(list(zip(x_list, y_list)), v_list)
        _ = self._space.reduce.Lambda._local_k0(interp, dg, degree, fc_range=[dest_index])
        cochain = _[dest_index]
        return cochain
