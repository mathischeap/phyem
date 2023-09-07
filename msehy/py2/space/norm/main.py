# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.space.norm.Lambda import MseHyPy2SpaceNormLambda


class MseHyPy2SpaceNorm(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._norm = None
        self._freeze()

    def __call__(self, cochain, quad_degree=None, **kwargs):
        """find the error at time `t`."""
        indicator = self._space.abstract.indicator
        if self._norm is None:
            if indicator == 'Lambda':
                self._norm = MseHyPy2SpaceNormLambda(self._space)
            else:
                raise NotImplementedError(f"{indicator}.")

        return self._norm(cochain, quad_degree, **kwargs)
