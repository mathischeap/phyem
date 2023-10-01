# -*- coding: utf-8 -*-
r"""
"""

from tools.frozen import Frozen
from generic.py._2d_unstruct.space.reconstruct.Lambda import ReconstructLambda


class Reconstruct(Frozen):
    """generation dependent"""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda = None
        self._freeze()

    def __call__(self, cochain, meshgrid_xi, meshgrid_et, ravel=False, element_range=None, degree=None):
        """Reconstruct using cochain at time `t`; generation dependent"""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            return self.Lambda(
                cochain, meshgrid_xi, meshgrid_et, ravel=ravel, element_range=element_range, degree=degree,
            )
        else:
            raise NotImplementedError(f"{indicator}.")

    @property
    def Lambda(self):
        if self._Lambda is None:
            self._Lambda = ReconstructLambda(self._space)
        return self._Lambda
