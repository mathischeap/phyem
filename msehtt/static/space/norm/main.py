# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from msehtt.static.space.norm.Lambda.main import MseHttSpaceNormLambda


class MseHttSpaceNorm(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, degree, cochain, norm_type='L2'):
        """"""
        indicator = self._space.indicator
        if indicator == 'Lambda':
            return MseHttSpaceNormLambda(self._space)(degree, cochain, norm_type=norm_type)
        else:
            raise NotImplementedError()
