# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msepy.space.incidence_matrix.Lambda import MsePyIncidenceMatrixLambda
from msepy.space.incidence_matrix.bundle import MsePyIncidenceMatrixBundle


class MsePyIncidenceMatrix(Frozen):
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

    @property
    def Lambda(self):
        if self._Lambda is None:
            self._Lambda = MsePyIncidenceMatrixLambda(self._space)
        return self._Lambda

    @property
    def bundle(self):
        if self._bundle is None:
            self._bundle = MsePyIncidenceMatrixBundle(self._space)
        return self._bundle
