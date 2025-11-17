# -*- coding: utf-8 -*-
"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.space.reconstruct.Lambda.main import MseHttSpaceReconstructLambda


class MseHttSpaceReconstruct(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, degree, cochain, *meshgrid, ravel=False, element_range=None):
        """reconstruct at ``meshgrid``."""
        indicator = self._space.indicator
        if indicator == 'Lambda':
            return MseHttSpaceReconstructLambda(self._space)(
                degree, cochain, *meshgrid, ravel=ravel, element_range=element_range)
        else:
            raise NotImplementedError()
