# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen

from msehtt.static.space.local_numbering.Lambda.main import MseHttSpaceLocalNumberingLambda


class MseHttSpaceLocalNumbering(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, etype, degree):
        """"""
        indicator = self._space.indicator
        if indicator == 'Lambda':
            return MseHttSpaceLocalNumberingLambda(self._space)(etype, degree)
        else:
            raise NotImplementedError()
