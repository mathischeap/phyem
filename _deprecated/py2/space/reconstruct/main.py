# -*- coding: utf-8 -*-
r"""
"""

from tools.frozen import Frozen
from msehy.py2.space.reconstruct.Lambda import MseHyPy2SpaceReconstructLambda


class MseHyPy2SpaceReconstruct(Frozen):
    """generation dependent"""

    def __init__(self, space):
        """"""
        self._space = space
        self._reconstruct = None
        self._freeze()

    def __call__(self, g, cochain, *meshgrid, **kwargs):
        """Reconstruct using cochain at time `t`; generation dependent"""
        if self._reconstruct is None:
            indicator = self._space.abstract.indicator
            if indicator == 'Lambda':
                self._reconstruct = MseHyPy2SpaceReconstructLambda(self._space)
            else:
                raise NotImplementedError(f"{indicator}.")

        return self._reconstruct(g, cochain, *meshgrid, **kwargs)
