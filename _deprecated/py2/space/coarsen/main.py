# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.space.coarsen.Lambda import MseHyPy2CoarsenLambda


class MseHyPy2Coarsen(Frozen):
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
            self._Lambda = MseHyPy2CoarsenLambda(self._space)
        return self._Lambda
