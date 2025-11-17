# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen

from phyem.msehtt.static.space.reduce.Lambda.main import MseHttSpaceReduceLambda


class MseHttSpaceReduce(Frozen):
    r""""""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, cf_t, degree):
        """Reduce target at time `t` to space of degree ``degree`` on partial mesh ``tpm`` """
        indicator = self._space.indicator
        if indicator == 'Lambda':
            return MseHttSpaceReduceLambda(self._space)(cf_t, degree)
        else:
            raise NotImplementedError()
