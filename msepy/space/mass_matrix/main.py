# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msepy.space.mass_matrix.Lambda import MsePyMassMatrixLambda
from phyem.msepy.space.mass_matrix.bundle import MsePyMassMatrixBundle


class MsePyMassMatrix(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda = None
        self._bundle = None
        self._freeze()

    def __call__(self, degree, quad=None):
        """"""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            return self.Lambda(degree, quad=quad)
        elif indicator == 'bundle':
            return self.bundle(degree, quad=quad)
        else:
            raise NotImplementedError()

    @property
    def Lambda(self):
        if self._Lambda is None:
            self._Lambda = MsePyMassMatrixLambda(self._space)
        return self._Lambda

    @property
    def bundle(self):
        if self._bundle is None:
            self._bundle = MsePyMassMatrixBundle(self._space)
        return self._bundle
