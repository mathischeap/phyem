# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHyPy2GatheringMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._mesh = space.mesh
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._orientation = space.abstract.orientation
        self._cache = dict()
        self._freeze()

    def __call__(self, degree, generation):
        """"""
        generation = self._mesh._pg(generation)
        p = self._space[degree].p
        cache_key = str(p)  # only need p
        cached, cache_gm = self._mesh.generations.sync_cache(self._cache, generation, cache_key)
        if cached:
            return cache_gm
        else:
            pass
        if self._n == 2 and self._k == 1:
            method_name = f"_n{self._n}_k{self._k}_{self._orientation}"
        else:
            method_name = f"_n{self._n}_k{self._k}"
        gm = getattr(self, method_name)(p)
        self._mesh.generations.sync_cache(self._cache, generation, cache_key, data=gm)
        return gm

    def _n2_k2(self, p):
        """"""
        raise NotImplementedError()
