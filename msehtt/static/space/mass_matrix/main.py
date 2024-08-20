# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehtt.static.space.mass_matrix.Lambda.main import MseHttSpaceLambdaMassMatrix


class MseHttSpaceMassMatrix(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, degree):
        """"""
        indicator = self._space.indicator
        if indicator == 'Lambda':
            mm, cache_key_dict = MseHttSpaceLambdaMassMatrix(self._space)(degree)
        else:
            raise NotImplementedError()
        return mm, cache_key_dict
