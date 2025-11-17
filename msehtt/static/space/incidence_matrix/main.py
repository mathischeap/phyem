# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.space.incidence_matrix.Lambda.main import MseHttSpaceLambdaIncidenceMatrix


class MseHttSpaceIncidenceMatrix(Frozen):
    r""""""

    def __init__(self, space):
        r""""""
        self._space = space
        self._freeze()

    def __call__(self, degree):
        r""""""
        indicator = self._space.indicator
        if indicator == 'Lambda':
            im, im_cache_key = MseHttSpaceLambdaIncidenceMatrix(self._space)(degree)
        else:
            raise NotImplementedError()
        return im, im_cache_key
