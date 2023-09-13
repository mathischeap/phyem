# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.space.reduce.Lambda import MseHyPy2SpaceReduceLambda


class MseHyPySpaceReduce(Frozen):
    """generation dependent"""

    def __init__(self, space):
        """"""
        self._space = space
        self._reduce = None
        self._Lambda = None
        self._freeze()

    def __call__(self, cf, t, g, degree, **kwargs):
        """generation dependent"""
        if self._reduce is None:
            indicator = self._space.abstract.indicator
            if indicator == 'Lambda':
                self._reduce = self.Lambda
            else:
                raise NotImplementedError(f"{indicator}.")

        return self._reduce(cf, t, g, degree, **kwargs)

    @property
    def Lambda(self):
        if self._Lambda is None:
            self._Lambda = MseHyPy2SpaceReduceLambda(self._space)
        return self._Lambda
