# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msepy.space.reduce.Lambda import MsePySpaceReduceLambda
from phyem.msepy.space.reduce.bundle import MsePySpaceReduceBundle


class MsePySpaceReduce(Frozen):
    r""""""

    def __init__(self, space):
        r""""""
        self._space = space
        self._reduce = None
        self._freeze()

    def __call__(self, cf, t, degree, **kwargs):
        r""""""
        if self._reduce is None:
            indicator = self._space.abstract.indicator
            if indicator == 'Lambda':
                self._reduce = MsePySpaceReduceLambda(self._space)
            elif indicator == 'bundle':
                self._reduce = MsePySpaceReduceBundle(self._space)
            elif indicator == 'bundle-diagonal':
                self._reduce = MsePySpaceReduceLambda(self._space)
            else:
                raise NotImplementedError(f"{indicator}.")

        return self._reduce(cf, t, degree, **kwargs)
