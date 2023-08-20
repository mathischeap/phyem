# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msepy.space.gathering_matrix.Lambda import MsePyGatheringMatrixLambda
from msepy.space.gathering_matrix.bundle import MsePyGatheringMatrixBundle


class MsePyGatheringMatrix(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda = None
        self._bundle = None
        self._freeze()

    def __call__(self, degree):
        """"""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            return self.Lambda(degree)
        elif indicator == 'bundle':
            return self.bundle(degree)
        else:
            raise NotImplementedError()

    def _next(self, degree):
        """The gathering matrix of d(space)."""
        from msepy.main import new
        ab_space = self._space.abstract
        d_ab_space = ab_space.d()
        d_msepy_space = new(d_ab_space)  # make msepy space, must using this function.
        return d_msepy_space.gathering_matrix(degree)

    @property
    def Lambda(self):
        if self._Lambda is None:
            self._Lambda = MsePyGatheringMatrixLambda(self._space)
        return self._Lambda

    @property
    def bundle(self):
        if self._bundle is None:
            self._bundle = MsePyGatheringMatrixBundle(self._space)
        return self._bundle
