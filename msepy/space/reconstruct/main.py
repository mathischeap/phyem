# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.space.reconstruct.Lambda import MsePySpaceReconstructLambda


class MsePySpaceReconstruct(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._reconstruct = None
        self._freeze()

    def __call__(self, local_cochain, degree, *meshgrid, **kwargs):
        """Reconstruct using cochain at time `t`."""
        if self._reconstruct is None:
            indicator = self._space.abstract.indicator
            if indicator == 'Lambda':
                self._reconstruct = MsePySpaceReconstructLambda(self._space)
            else:
                raise NotImplementedError(f"{indicator}.")

        return self._reconstruct(local_cochain, degree, *meshgrid, **kwargs)
