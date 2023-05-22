# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
from tools.frozen import Frozen
from msepy.space.basis_functions.Lambda import MsePyBasisFunctionsLambda


class MsePyBasisFunctions(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda_cache = {}
        self._freeze()

    def __getitem__(self, degree):
        """Return"""
        indicator = self._space.abstract.indicator
        key = str(degree)
        if indicator == 'Lambda':
            if key in self._Lambda_cache:
                Lambda_bf = self._Lambda_cache[key]
            else:
                Lambda_bf = MsePyBasisFunctionsLambda(self._space, degree)
                self._Lambda_cache[key] = Lambda_bf
            return Lambda_bf
        else:
            raise NotImplementedError()
