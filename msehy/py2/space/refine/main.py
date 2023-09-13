# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.space.refine.Lambda import MseHyPy2RefineLambda


class MseHyPy2Refine(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda = None
        self._freeze()

    def __call__(self, *args, **kwargs):
        """"""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            return self.Lambda(*args, **kwargs)
        else:
            raise NotImplementedError

    @property
    def Lambda(self):
        if self._Lambda is None:
            self._Lambda = MseHyPy2RefineLambda(self._space)
        return self._Lambda
