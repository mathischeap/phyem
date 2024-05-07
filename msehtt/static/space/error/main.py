# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from msehtt.static.space.error.Lambda.main import MseHttSpaceLambdaError


class MseHttSpaceError(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, cf, cochain, degree, error_type):
        """"""
        indicator = self._space.indicator
        if indicator == 'Lambda':
            return MseHttSpaceLambdaError(self._space)(cf, cochain, degree, error_type)
        else:
            raise NotImplementedError()
