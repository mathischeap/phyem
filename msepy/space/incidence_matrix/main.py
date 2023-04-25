# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
import sys
if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.space.incidence_matrix.Lambda import MsePyIncidenceMatrixLambda


class MsePyIncidenceMatrix(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, degree):
        """"""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            return MsePyIncidenceMatrixLambda(self._space)(degree)
        else:
            raise NotImplementedError()
