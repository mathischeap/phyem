# -*- coding: utf-8 -*-
"""
"""
from phyem.tools.frozen import Frozen
from phyem.msepy.space.wedge_matrix.Lambda import MsePyWedgeMatrixLambda


class MsePyWedgeMatrix(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda = None
        self._bundle = None
        self._freeze()

    def __call__(self, other_space, self_degree, other_degree, quad=None):
        """"""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            return self.Lambda(other_space, self_degree, other_degree, quad=quad)
        else:
            raise NotImplementedError()

    @property
    def Lambda(self):
        if self._Lambda is None:
            self._Lambda = MsePyWedgeMatrixLambda(self._space)
        return self._Lambda
