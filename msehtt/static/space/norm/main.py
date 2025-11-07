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

    def __call__(self, degree, cochain, norm_type='L2', component_wise=False):
        """"""
        indicator = self._space.indicator
        if indicator == 'Lambda':
            return MseHttSpaceNormLambda(self._space)(
                degree, cochain, norm_type=norm_type, component_wise=component_wise
            )
        else:
            raise NotImplementedError()
