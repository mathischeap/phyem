# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen

from msehtt.static.space.reduce.Lambda.main import MseHttSpaceReduceLambda


class MseHttSpaceReduce(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, target, t, tpm, degree):
        """Reduce target at time `t` to space of degree ``degree`` on partial mesh ``tpm`` """
        indicator = self._space.indicator
        if indicator == 'Lambda':
            return MseHttSpaceReduceLambda(self._space)(target, t, tpm, degree)
        else:
            raise NotImplementedError()
