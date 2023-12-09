# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from legacy.generic.py._2d_unstruct.space.norm.Lambda import NormLambda


class Norm(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._norm = None
        self._freeze()

    def __call__(self, cochain, d=2):
        """find the error at time `t`."""
        indicator = self._space.abstract.indicator
        if self._norm is None:
            if indicator == 'Lambda':
                self._norm = NormLambda(self._space)
            else:
                raise NotImplementedError(f"{indicator}.")

        return self._norm(cochain, d=d)
