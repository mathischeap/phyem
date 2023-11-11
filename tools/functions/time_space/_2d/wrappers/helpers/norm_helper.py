# -*- coding: utf-8 -*-
r"""
"""
import numpy as np


class NormHelper2DVector:
    """"""

    def __init__(self, v0, v1):
        """"""
        self._v0 = v0
        self._v1 = v1

    def __call__(self, t, x, y):
        return np.sqrt(self._v0(t, x, y) ** 2 + self._v1(t, x, y) ** 2)
