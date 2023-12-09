# -*- coding: utf-8 -*-
r"""
"""
from tools.quadrature import Quadrature
import numpy as np
from tools.frozen import Frozen
# from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator


class Coarsen(Frozen):
    """"""

    def __init__(self, space):
        self._space = space
        self._indicator = self._space.abstract.indicator
        self._sampling_cache = {}
        self._freeze()

    def _make_sampling(self, degree):
        if degree in self._sampling_cache:
            return self._sampling_cache[degree]
        else:
            p = self._space.generic[degree].p
            nodes = Quadrature(p+12, category='Gauss').quad_nodes
            sampling = np.linspace(nodes[0], nodes[-1], p+50)
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

    def _Lambda_k2_interp(self, degree, dest_index, source_indices, local_source_cochains):
        """"""
        sampling = self._make_sampling(degree)
        xy, v = self._space.previous.reconstruct(
            local_source_cochains, sampling, sampling, True, element_range=source_indices, degree=degree,
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
        _ = self._space.generic.reduce.Lambda._k2_local(interp, degree, element_range=[dest_index])
        cochain = _[dest_index]
        return cochain

    def _Lambda_k0(self, *args, method='interp'):

        if method == 'interp':
            return self._Lambda_k0_interp(*args)
        else:
            raise NotImplementedError()

    def _Lambda_k0_interp(self, degree, dest_index, source_indices, local_source_cochains):
        """"""
        sampling = self._make_sampling(degree)
        xy, v = self._space.previous.reconstruct(
            local_source_cochains, sampling, sampling, True, element_range=source_indices, degree=degree,
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
        _ = self._space.generic.reduce.Lambda._k0_local(interp, degree, element_range=[dest_index])
        cochain = _[dest_index]
        return cochain

    def _Lambda_k1_inner(self, *args, method='interp'):

        if method == 'interp':
            return self._Lambda_k1_inner_interp(*args)
        else:
            raise NotImplementedError()

    def _Lambda_k1_inner_interp(self, degree, dest_index, source_indices, local_source_cochains):
        """"""
        sampling = self._make_sampling(degree)
        xy, v = self._space.previous.reconstruct(
            local_source_cochains, sampling, sampling, True, element_range=source_indices, degree=degree,
        )
        x, y = xy
        u, v = v
        x_list = list()
        y_list = list()
        u_list = list()
        v_list = list()
        for index in source_indices:
            x_list.extend(x[index])
            y_list.extend(y[index])
            u_list.extend(u[index])
            v_list.extend(v[index])
        xy = list(zip(x_list, y_list))
        interp_x = NearestNDInterpolator(xy, u_list)
        interp_y = NearestNDInterpolator(xy, v_list)
        _ = self._space.generic.reduce.Lambda._k1_inner_local(
            interp_x, interp_y, degree, element_range=[dest_index])
        cochain = _[dest_index]
        return cochain

    def _Lambda_k1_outer(self, *args, method='interp'):

        if method == 'interp':
            return self._Lambda_k1_outer_interp(*args)
        else:
            raise NotImplementedError()

    def _Lambda_k1_outer_interp(self, degree, dest_index, source_indices, local_source_cochains):
        """"""
        sampling = self._make_sampling(degree)
        xy, v = self._space.previous.reconstruct(
            local_source_cochains, sampling, sampling, True, element_range=source_indices, degree=degree,
        )
        x, y = xy
        u, v = v
        x_list = list()
        y_list = list()
        u_list = list()
        v_list = list()
        for index in source_indices:
            x_list.extend(x[index])
            y_list.extend(y[index])
            u_list.extend(u[index])
            v_list.extend(v[index])
        xy = list(zip(x_list, y_list))
        interp_x = NearestNDInterpolator(xy, u_list)
        interp_y = NearestNDInterpolator(xy, v_list)
        _ = self._space.generic.reduce.Lambda._k1_outer_local(
            interp_x, interp_y, degree, element_range=[dest_index])
        cochain = _[dest_index]
        return cochain
