# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.space.num_local_dofs.Lambda import MseHyPy2NumLocalDofsLambda


class MseHyPy2NumLocalDofs(Frozen):
    """generation in-dependent."""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda = None
        self._freeze()

    def __call__(self, degree):
        """"""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            return self.Lambda(degree)
        else:
            raise NotImplementedError()

    @property
    def Lambda(self):
        if self._Lambda is None:
            self._Lambda = MseHyPy2NumLocalDofsLambda(self._space)
        return self._Lambda
