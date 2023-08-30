# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHyPy2MeshElementsLevelElementsCT(Frozen):
    """Not a level ct. Call me from ``level.elements``."""

    def __init__(self, level_elements):
        """"""
        self._level_elements = level_elements
        self._freeze()
