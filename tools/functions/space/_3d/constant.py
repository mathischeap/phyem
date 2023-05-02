# -*- coding: utf-8 -*-
import numpy as np
from abc import ABC


class CFG(ABC):
    """Constant function generator."""
    def __init__(self, C):
        """ """
        self._C_ = C

    def _constant_func_(self, x, y, z):
        assert np.shape(x) == np.shape(y) == np.shape(z)
        return self._C_ + np.zeros(np.shape(x))

    def __call__(self):
        return self._constant_func_
