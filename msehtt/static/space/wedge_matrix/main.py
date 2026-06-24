# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.space.wedge_matrix.Lambda.main import MseHtt_SpaceLambda_Wedge_Matrix


class MseHttSpace_Wedge_Matrix(Frozen):
    r""""""

    def __init__(self, space):
        r""""""
        self._space = space
        self._freeze()

    def __call__(self, degree0, space1, degree1):
        r""""""
        space0 = self._space
        indicator0 = space0.indicator
        indicator1 = space1.indicator

        assert space0.tpm is space1.tpm

        if indicator0 == indicator1 == 'Lambda':
            W, cache_key_dict = MseHtt_SpaceLambda_Wedge_Matrix(space0, space1)(degree0, degree1)
        else:
            raise NotImplementedError()
        return W, cache_key_dict
