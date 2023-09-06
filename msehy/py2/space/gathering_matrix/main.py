# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.space.gathering_matrix.Lambda import MseHyPy2GatheringMatrixLambda


class MseHyPy2GatheringMatrix(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._mesh = space.mesh
        self._Lambda = None
        self._freeze()

    def __call__(self, degree, generation=None):
        """"""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            return self.Lambda(degree, generation)
        else:
            raise NotImplementedError()

    def _next(self, degree):
        """The gathering matrix of d(space)."""
        from msehy.py2.main import new
        ab_space = self._space.abstract
        d_ab_space = ab_space.d()
        d_msepy_space = new(d_ab_space)  # make msepy space, must using this function.
        return d_msepy_space.gathering_matrix(degree)

    @property
    def Lambda(self):
        if self._Lambda is None:
            self._Lambda = MseHyPy2GatheringMatrixLambda(self._space)
        return self._Lambda
