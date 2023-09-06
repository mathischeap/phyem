# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHyPy2LocalNumberingLambda(Frozen):
    """Generation independent."""

    def __init__(self, space):
        """Generation independent."""
        self._space = space
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._orientation = space.abstract.orientation
        self._freeze()

    def __call__(self, degree):
        """Generation independent."""
        p = self._space[degree].p
        if self._n == 2 and self._k == 1:
            method_name = f"_n{self._n}_k{self._k}_{self._orientation}"
        else:
            method_name = f"_n{self._n}_k{self._k}"
        ln = getattr(self, method_name)(p)
        return ln

    def _n2_k2(self, p):
        """Generation independent."""
        ln = {
            'q': 1,  # for quadrilateral elements
            't': 1,  # triangle elements
        }
        raise NotImplementedError()
