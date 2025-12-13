# -*- coding: utf-8 -*-
"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.space.mass.Lambda.main import MseHttSpace_MASS_Lambda


class MseHttSpace_MASS(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, degree, cochain,):
        """"""
        indicator = self._space.indicator
        if indicator == 'Lambda':
            return MseHttSpace_MASS_Lambda(self._space)(
                degree, cochain,
            )
        else:
            raise NotImplementedError()
