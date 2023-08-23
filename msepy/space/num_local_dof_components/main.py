# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msepy.space.num_local_dof_components.Lambda import MsePyNumLocalDofComponentsLambda
from msepy.space.num_local_dof_components.bundle import MsePyNumLocalDofComponentsBundle


class MsePyNumLocalDofComponents(Frozen):
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
        elif indicator == 'bundle-diagonal':
            return self.Lambda(degree)
        else:
            raise NotImplementedError()

    @property
    def Lambda(self):
        if self._Lambda is None:
            self._Lambda = MsePyNumLocalDofComponentsLambda(self._space)
        return self._Lambda

    @property
    def bundle(self):
        if self._bundle is None:
            self._bundle = MsePyNumLocalDofComponentsBundle(self._space)
        return self._bundle
