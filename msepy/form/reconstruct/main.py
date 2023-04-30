# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.form.reconstruct.Lambda import MsePyReconstructLambda


class MsePyRootFormReconstruct(Frozen):
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
            reconstruct = MsePyReconstructLambda(self._f, t)
        else:
            raise NotImplementedError(f"{indicator}.")

        return reconstruct
