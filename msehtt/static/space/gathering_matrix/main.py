# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen

from msehtt.static.space.gathering_matrix.Lambda.main import MseHttSpaceGatheringMatrixLambda


class MseHttSpaceGatheringMatrix(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, tpm, degree):
        """"""
        indicator = self._space.indicator
        if indicator == 'Lambda':
            return MseHttSpaceGatheringMatrixLambda(self._space)(tpm, degree)
        else:
            raise NotImplementedError()
