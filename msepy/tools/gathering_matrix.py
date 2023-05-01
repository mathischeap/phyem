# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
"""

from tools.frozen import Frozen


class RegularGatheringMatrix(Frozen):
    """2-d-array: 0-axis -> #elements, 1-axis: local-numbering, values -> global-numbering."""

    def __init__(self, _2d_array):
        """"""
        self._gathering_matrix = _2d_array
        self._freeze()

    def __getitem__(self, i):
        """Return the global_numbering for dofs in element #i."""
        return self._gathering_matrix[i]

    @property
    def shape(self):
        """the shape of the 2d array gm."""
        return self._gathering_matrix.shape