# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msepy.space.num_local_dofs.Lambda import MsePyNumLocalDofsLambda
from phyem.msepy.space.num_local_dofs.bundle import MsePyNumLocalDofsBundle


class MsePyNumLocalDofs(Frozen):
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
            self._Lambda = MsePyNumLocalDofsLambda(self._space)
        return self._Lambda

    @property
    def bundle(self):
        if self._bundle is None:
            self._bundle = MsePyNumLocalDofsBundle(self._space)
        return self._bundle
