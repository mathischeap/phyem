# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.frozen import Frozen


class ___EXP_HELPER___(Frozen):
    r""""""

    def __init__(self, func):
        self._f = func
        self._freeze()

    def __call__(self, t, x, y):
        r"""Compute exp^{ func(t, x, y) }"""
        return np.exp(self._f(t, x, y))
