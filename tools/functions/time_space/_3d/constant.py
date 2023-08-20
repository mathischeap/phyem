# -*- coding: utf-8 -*-
r"""
4D (t, x, y, z) constant function.
"""
from abc import ABC


class CFG(ABC):
    """Constant function generator."""
    def __init__(self, C):
        """ """
        self._C_ = C

    def _constant_func_(self, t, x, y, z):
        """ """
        return self._C_ + 0 * t * x * y * z

    def __call__(self):
        """ """
        return self._constant_func_
