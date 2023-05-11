# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
import sys
if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.space.gathering_matrix.Lambda import MsePyGatheringMatrixLambda


class MsePyGatheringMatrix(Frozen):
    """"""

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

    def _next(self, degree):
        """The gathering matrix of d(space)."""
        from msepy.main import new
        ab_space = self._space.abstract
        d_ab_space = ab_space.d()
        d_msepy_space = new(d_ab_space)  # make msepy space, must using this function.
        return d_msepy_space.gathering_matrix(degree)

    @property
    def Lambda(self):
        if self._Lambda is None:
            self._Lambda = MsePyGatheringMatrixLambda(self._space)
        return self._Lambda
