# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 4:13 PM on 5/1/2023
"""
from tools.frozen import Frozen
from msepy.form.error.Lambda import MsePyRootFormErrorLambda


class MsePyRootFormError(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._freeze()

    def __getitem__(self, t):
        """Reconstruct using cochain at time `t`."""
        space = self._f.space
        indicator = space.abstract.indicator
        if indicator == 'Lambda':
            reconstruct = MsePyRootFormErrorLambda(self._f, t)
        else:
            raise NotImplementedError(f"{indicator}.")

        return reconstruct
