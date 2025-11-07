# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen


class ___LOG_HELPER___(Frozen):
    r""""""

    def __init__(self, func, base=np.e):
        self._f = func
        self._b = base
        self._freeze()

    def __call__(self, t, x, y):
        r"""Compute log_{base} func(t, x, y)"""
        f = self._f(t, x, y)
        if self._b == np.e:
            return np.log(f)
        else:
            raise NotImplementedError()
