# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehtt.static.space.local_dofs.Lambda.main import MseHttSpace_Local_Dofs_Lambda


class MseHttSpace_Local_Dofs(Frozen):
    r""""""

    def __init__(self, space):
        r""""""
        self._space = space
        self._freeze()

    def __call__(self, degree):
        r""""""
        indicator = self._space.indicator
        if indicator == 'Lambda':
            LG = MseHttSpace_Local_Dofs_Lambda(self._space)(degree)
        else:
            raise NotImplementedError()
        return LG
