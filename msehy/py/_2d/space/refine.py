# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.quadrature import Quadrature
from tools.frozen import Frozen
from scipy.interpolate import NearestNDInterpolator
from tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer
from src.spaces.main import _degree_str_maker


class Refine(Frozen):
    """"""

    def __init__(self, space):
        self._space = space
        self._indicator = self._space.abstract.indicator
        self._sampling_cache = {}
        self._cache_0 = {}
        self._cache_2 = {}
        self._cache_1_inner = {}
        self._cache_1_outer = {}
        self._freeze()

    def _make_sampling(self, degree):
        if degree in self._sampling_cache:
            return self._sampling_cache[degree]
        else:
            p = self._space.generic[degree].p
            nodes = Quadrature(p+15, category='Gauss').quad_nodes
            sampling = np.linspace(nodes[0], nodes[-1], p+75)
            self._sampling_cache[degree] = sampling
            return sampling

    def __call__(self, *args, method='interp'):
        """"""
        if self._indicator == 'Lambda':
            k = self._space.abstract._k
            orientation = self._space.abstract.orientation
            if k == 2:
                return self._Lambda_k2(*args, method=method)
            elif k == 0:
                return self._Lambda_k0(*args, method=method)
            elif k == 1 and orientation == 'inner':
                return self._Lambda_k1_inner(*args, method=method)
            elif k == 1 and orientation == 'outer':
                return self._Lambda_k1_outer(*args, method=method)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def _Lambda_k2(self, *args, method='interp'):

        if method == 'interp':
            return self._Lambda_k2_interp(*args)
        else:
            raise NotImplementedError()

    def _Lambda_k2_interp(self, degree, dest_index, source_index, local_source_cochain):
        """"""
        check_str = f"{_degree_str_maker(degree)}-{source_index}-{self._space.generation}"
        cached, interp = ndarray_key_comparer(self._cache_2, [local_source_cochain], check_str=check_str)
        if cached:
            pass
        else:
            sampling = self._make_sampling(degree)
            source_space = self._space.previous
            xy, v = source_space.reconstruct(
                {source_index: local_source_cochain},
                sampling, sampling,
                True,
                element_range=[source_index],
                degree=degree,
            )
            x, y = xy
            v = v[0]
            x = x[source_index]
            y = y[source_index]
            v = v[source_index]
            interp = NearestNDInterpolator(list(zip(x, y)), v)

            add_to_ndarray_cache(
                self._cache_2,
                [local_source_cochain],
                interp,
                check_str=check_str, maximum=8
            )

        _ = self._space.generic.reduce.Lambda._k2_local(interp, degree, element_range=[dest_index])
        cochain = _[dest_index]
        return cochain

    def _Lambda_k0(self, *args, method='interp'):

        if method == 'interp':
            return self._Lambda_k0_interp(*args)
        else:
            raise NotImplementedError()

    def _Lambda_k0_interp(self, degree, dest_index, source_index, local_source_cochain):
        """"""
        check_str = f"{_degree_str_maker(degree)}-{source_index}-{self._space.generation}"
        cached, interp = ndarray_key_comparer(self._cache_0, [local_source_cochain], check_str=check_str)
        if cached:
            pass
        else:
            sampling = self._make_sampling(degree)
            source_space = self._space.previous
            xy, v = source_space.reconstruct(
                {source_index: local_source_cochain},
                sampling, sampling,
                True,
                element_range=[source_index],
                degree=degree,
            )
            x, y = xy
            v = v[0]
            x = x[source_index]
            y = y[source_index]
            v = v[source_index]
            interp = NearestNDInterpolator(list(zip(x, y)), v)

            add_to_ndarray_cache(
                self._cache_0,
                [local_source_cochain],
                interp,
                check_str=check_str, maximum=8
            )

        _ = self._space.generic.reduce.Lambda._k0_local(interp, degree, element_range=[dest_index])
        cochain = _[dest_index]
        return cochain

    def _Lambda_k1_inner(self, *args, method='interp'):

        if method == 'interp':
            return self._Lambda_k1_inner_interp(*args)
        else:
            raise NotImplementedError()

    def _Lambda_k1_inner_interp(self, degree, dest_index, source_index, local_source_cochain):
        """"""
        check_str = f"{_degree_str_maker(degree)}-{source_index}-{self._space.generation}"
        cached, interp = ndarray_key_comparer(self._cache_1_inner, [local_source_cochain], check_str=check_str)
        if cached:
            interp_x, interp_y = interp
        else:
            sampling = self._make_sampling(degree)
            source_space = self._space.previous
            xy, v = source_space.reconstruct(
                {source_index: local_source_cochain},
                sampling, sampling,
                True,
                element_range=[source_index],
                degree=degree,
            )
            x, y = xy
            u, v = v
            x = x[source_index]
            y = y[source_index]
            u = u[source_index]
            v = v[source_index]
            xy = list(zip(x, y))
            interp_x = NearestNDInterpolator(xy, u)
            interp_y = NearestNDInterpolator(xy, v)

            add_to_ndarray_cache(
                self._cache_1_inner,
                [local_source_cochain],
                (interp_x, interp_y),
                check_str=check_str, maximum=8
            )

        _ = self._space.generic.reduce.Lambda._k1_inner_local(
            interp_x, interp_y, degree, element_range=[dest_index])
        cochain = _[dest_index]
        return cochain

    def _Lambda_k1_outer(self, *args, method='interp'):

        if method == 'interp':
            return self._Lambda_k1_outer_interp(*args)
        else:
            raise NotImplementedError()

    def _Lambda_k1_outer_interp(self, degree, dest_index, source_index, local_source_cochain):
        """"""
        check_str = f"{_degree_str_maker(degree)}-{source_index}-{self._space.generation}"
        cached, interp = ndarray_key_comparer(self._cache_1_outer, [local_source_cochain], check_str=check_str)
        if cached:
            interp_x, interp_y = interp
        else:
            sampling = self._make_sampling(degree)
            source_space = self._space.previous
            xy, v = source_space.reconstruct(
                {source_index: local_source_cochain},
                sampling, sampling,
                True,
                element_range=[source_index],
                degree=degree,
            )
            x, y = xy
            u, v = v
            x = x[source_index]
            y = y[source_index]
            u = u[source_index]
            v = v[source_index]
            xy = list(zip(x, y))
            interp_x = NearestNDInterpolator(xy, u)
            interp_y = NearestNDInterpolator(xy, v)

            add_to_ndarray_cache(
                self._cache_1_outer,
                [local_source_cochain],
                (interp_x, interp_y),
                check_str=check_str, maximum=8
            )

        _ = self._space.generic.reduce.Lambda._k1_outer_local(
            interp_x, interp_y, degree, element_range=[dest_index])
        cochain = _[dest_index]
        return cochain
