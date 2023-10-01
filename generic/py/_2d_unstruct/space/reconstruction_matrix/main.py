# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from generic.py._2d_unstruct.space.reconstruction_matrix.Lambda import ReconstructMatrixLambda


class ReconstructMatrix(Frozen):
    """Build reconstruct matrix for particular elements."""

    def __init__(self, space):
        """"""
        self._space = space
        self._Lambda = None
        self._freeze()

    def __call__(self, degree, xi, et, element_range=None):
        """"""
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            return self.Lambda(degree, xi, et, element_range=element_range)
        else:
            raise NotImplementedError()

    @property
    def Lambda(self):
        """for scalar valued form spaces."""
        if self._Lambda is None:
            self._Lambda = ReconstructMatrixLambda(self._space)
        return self._Lambda
