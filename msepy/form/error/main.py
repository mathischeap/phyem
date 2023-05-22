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
        """find the error at time `t`."""
        if t is None:
            t = self._f.cochain.newest
            assert t is not None, f"I have now newest cochain time!"
        else:
            pass
        space = self._f.space
        indicator = space.abstract.indicator
        if indicator == 'Lambda':
            error = MsePyRootFormErrorLambda(self._f, t)
        else:
            raise NotImplementedError(f"{indicator}.")

        return error
