# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
"""
import numpy as np
from tools.frozen import Frozen


class RegularGatheringMatrix(Frozen):
    """2-d-array: 0-axis -> #elements, 1-axis: local-numbering, values -> global-numbering."""

    def __init__(self, _2d_array):
        """"""
        self._gm = _2d_array
        self._num_dofs = None
        self._freeze()

    def __getitem__(self, i):
        """Return the global_numbering for dofs in element #i."""
        return self._gm[i]

    def __len__(self):
        """How many elements this gathering_matrix is representing?

        Same to `num_elements`.
        """
        return self.num_elements

    @property
    def shape(self):
        """the shape of the 2d array gm.

        Raise Error for irregular gathering matrices.
        """
        return self._gm.shape

    @property
    def num_dofs(self):
        if self._num_dofs is None:
            self._num_dofs = int(np.max(self._gm) + 1)
        return self._num_dofs

    @property
    def num_elements(self):
        """How many elements this gathering_matrix is representing?

        Same to `len`.
        """
        return self.shape[0]
