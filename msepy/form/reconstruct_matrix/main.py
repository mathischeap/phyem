# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 3:11 PM on 7/17/2023
"""
from tools.frozen import Frozen
from msepy.form.reconstruct_matrix.Lambda import MsePyMeshElementReconstructMatrixLambda


class MsePyMeshElementReconstructMatrix(Frozen):
    """Build reconstruct matrix for particular elements."""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._space = rf.space
        self._reconstruct = None
        self._freeze()

    def __call__(self, *xi_et_sg, element_range=None):
        """

        Parameters
        ----------
        xi_et_sg
        elements :
            If elements is None, we do it over all all elements.

        Returns
        -------

        """
        if self._reconstruct is None:
            indicator = self._f.space.abstract.indicator
            if indicator == 'Lambda':
                self._reconstruct = MsePyMeshElementReconstructMatrixLambda(self._f)
            else:
                raise NotImplementedError(f"{indicator}.")

        return self._reconstruct(*xi_et_sg, element_range=element_range)
