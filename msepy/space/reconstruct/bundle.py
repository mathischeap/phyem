# -*- coding: utf-8 -*-
r"""
"""

import numpy as np

from tools.frozen import Frozen


class MsePySpaceReconstructBundle(Frozen):
    """Reconstruct over all mesh-elements."""

    def __init__(self, space):
        """"""

        self._freeze()
