# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.space.local_numbering.Lambda import MseHyPy2LocalNumberingLambda


class MseHyPy2LocalNumbering(Frozen):
    """Generation independent."""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda = None
        self._freeze()

    def __call__(self, degree):
        """Generation independent."""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            return self.Lambda(degree)
        else:
            raise NotImplementedError()

    @property
    def Lambda(self):
        if self._Lambda is None:
            self._Lambda = MseHyPy2LocalNumberingLambda(self._space)
        return self._Lambda
