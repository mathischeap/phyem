# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from legacy.generic.py._2d_unstruct.space.reduce.Lambda import ReduceLambda


class Reduce(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._indicator = space.abstract.indicator
        self._Lambda = ReduceLambda(space)
        self._freeze()

    def __call__(self, target, t, degree):
        """"""
        if self._indicator == 'Lambda':
            return self.Lambda(target, t, degree)
        else:
            raise NotImplementedError()

    @property
    def Lambda(self):
        return self._Lambda