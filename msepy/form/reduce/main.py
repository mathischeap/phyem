# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.form.reduce.Lambda import MsePyReduceLambda


class MsePyRootFormReduce(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._reduce = None
        self._freeze()

    def __call__(self, t, update_cochain=True):
        """"""
        space = self._f.space
        indicator = space.abstract.indicator
        if indicator == 'Lambda':
            self._reduce = MsePyReduceLambda(self._f)
        else:
            raise NotImplementedError(f"{indicator}.")

        return self._reduce(t, update_cochain=update_cochain)
