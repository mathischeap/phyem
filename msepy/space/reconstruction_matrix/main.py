# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msepy.space.reconstruction_matrix.Lambda import MsePyReconstructMatrixLambda
from msepy.space.reconstruction_matrix.bundle import MsePyReconstructMatrixBundle


class MsePyReconstructMatrix(Frozen):
    """Build reconstruct matrix for particular elements."""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda = None
        self._bundle = None
        self._freeze()

    def __call__(self, degree, *meshgrid_xi_et_sg, element_range=None):
        """"""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            return self.Lambda(degree, *meshgrid_xi_et_sg, element_range=element_range)
        elif indicator == 'bundle':
            return self.bundle(degree, *meshgrid_xi_et_sg, element_range=element_range)
        else:
            raise NotImplementedError()

    @property
    def Lambda(self):
        """for scalar valued form spaces."""
        if self._Lambda is None:
            self._Lambda = MsePyReconstructMatrixLambda(self._space)
        return self._Lambda

    @property
    def bundle(self):
        """for scalar valued form spaces."""
        if self._bundle is None:
            self._bundle = MsePyReconstructMatrixBundle(self._space)
        return self._bundle
