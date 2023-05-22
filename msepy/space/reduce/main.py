# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""
from tools.frozen import Frozen
from msepy.space.reduce.Lambda import MsePySpaceReduceLambda


class MsePySpaceReduce(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._reduce = None
        self._freeze()

    def __call__(self, cf, t, degree, **kwargs):
        """"""
        if self._reduce is None:
            indicator = self._space.abstract.indicator
            if indicator == 'Lambda':
                self._reduce = MsePySpaceReduceLambda(self._space)
            else:
                raise NotImplementedError(f"{indicator}.")

        return self._reduce(cf, t, degree, **kwargs)
