# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from scipy.interpolate import NearestNDInterpolator
from tools.miscellaneous.ndarray_cache import ndarray_key_comparer, add_to_ndarray_cache
from src.spaces.main import _degree_str_maker


class MseHyPy2RefineLambda(Frozen):
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
        self._interp0_cache = dict()
        self._freeze()

    def __call__(self, degree, dest_g__dest_index, source_g__source_index__cochain):
        """"""
        dest_g, dest_index = dest_g__dest_index
        source_g, source_index, source_cochain = source_g__source_index__cochain
        assert dest_g != source_g, f"source g must be different from dest g."
        assert isinstance(source_cochain, np.ndarray), f"source_cochain of one cell must be a ndarray."

        if self._k == 0:
            return self._k0(
                degree, dest_index, source_index, source_cochain, (dest_g, source_g))
        # elif self._k == 1 and self._orientation == 'inner':
        #     return self._k1_inner(
        #         degree, dest_index, source_index, source_cochain, (dest_g, source_g))
        # elif self._k == 1 and self._orientation == 'outer':
        #     return self._k1_outer(
        #         degree, dest_index, source_index, source_cochain, (dest_g, source_g))
        # elif self._k == 2:
        #     return self._k2(
        #         degree, dest_index, source_index, source_cochain, (dest_g, source_g))
        else:
            raise Exception

    def _k0(self, degree, dest_index, source_index, source_cochain, dg_sg, scheme='interpolation'):
        """refine scalar valued 0-forms to `d_fc` from `s_fc` with cochain `source_cochain`"""

        if scheme == 'interpolation':
            cochain, error = self._k0_interpolation(degree, dest_index, source_index, source_cochain, dg_sg)
        else:
            raise NotImplementedError(f"scheme={scheme} not implemented for 0-form coarsen")
        return cochain, error

    def _parse_p_for_mesh_grid_0(self, degree):
        """"""
        p = self._space[degree].p
        if p in self._cache_p0_:
            pass
        else:
            px, py = p
            assert px == py
            self._cache_p0_[p] = np.linspace(-1, 1, 10*px + 50)
        return self._cache_p0_[p]

    def _k0_interpolation(self, degree, dest_index, source_index, source_cochain, dg_sg):
        """"""
        dg, sg = dg_sg
        check_str = _degree_str_maker(degree) + str(sg) + str(source_index)
        cached, interp = ndarray_key_comparer(self._interp0_cache, [source_cochain], check_str=check_str)
        if cached:
            pass
        else:
            _ = {source_index: source_cochain}
            xi_and_et = self._parse_p_for_mesh_grid_0(degree)
            xy, v = self._space.reconstruct(
                sg, _, xi_and_et, xi_and_et, ravel=True,
                fc_range=[source_index], degree=degree
            )
            x, y = xy
            v = v[0]
            x = x[source_index]
            y = y[source_index]
            v = v[source_index]
            interp = NearestNDInterpolator(list(zip(x, y)), v)
            add_to_ndarray_cache(self._interp0_cache, [source_cochain], interp, check_str=check_str, maximum=8)

        _ = self._space.reduce.Lambda._local_k0(interp, dg, degree, fc_range=[dest_index])
        cochain = _[dest_index]
        error = 0
        return cochain, error
