# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen

from msehtt.static.space.reconstruction_matrix.Lambda.main import MseHttSpaceReconstructMatrixLambda


class MseHttSpaceReconstructionMatrix(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, degree, *meshgrid):
        """reconstruct at ``meshgrid``."""
        indicator = self._space.indicator
        if indicator == 'Lambda':
            return MseHttSpaceReconstructMatrixLambda(self._space)(degree, *meshgrid)
        else:
            raise NotImplementedError()
