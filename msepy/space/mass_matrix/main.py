# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
import sys
if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.space.mass_matrix.Lambda import MsePyMassMatrixLambda


class MsePyMassMatrix(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda = None
        self._freeze()

    def __call__(self, degree, quad_degree=None):
        """"""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            return self.Lambda(degree, quad_degree=quad_degree)
        else:
            raise NotImplementedError()

    @property
    def Lambda(self):
        if self._Lambda is None:
            self._Lambda = MsePyMassMatrixLambda(self._space)
        return self._Lambda
