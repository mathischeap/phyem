# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
import sys
if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.space.basic_functions.Lambda import MsePyBasicFunctionsLambda


class MsePyBasicFunctions(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda = None
        self._freeze()

    def __getitem__(self, degree):
        """Return"""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            self.Lambda._set_degree(degree)
            return self.Lambda
        else:
            raise NotImplementedError()

    @property
    def Lambda(self):
        if self._Lambda is None:
            self._Lambda = MsePyBasicFunctionsLambda(self._space)
        return self._Lambda
