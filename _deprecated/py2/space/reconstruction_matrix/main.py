# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.space.reconstruction_matrix.Lambda import MseHyPy2ReconstructMatrixLambda


class MseHyPy2ReconstructMatrix(Frozen):
    """Build reconstruct matrix for particular elements."""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda = None
        self._bundle = None
        self._freeze()

    def __call__(self, degree, g, *meshgrid_xi_et, fc_range=None):
        """"""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            return self.Lambda(degree, g, *meshgrid_xi_et, fc_range=fc_range)
        else:
            raise NotImplementedError()

    @property
    def Lambda(self):
        """for scalar valued form spaces."""
        if self._Lambda is None:
            self._Lambda = MseHyPy2ReconstructMatrixLambda(self._space)
        return self._Lambda
