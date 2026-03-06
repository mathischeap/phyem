# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.frozen import Frozen


class ___LOG_HELPER_3___(Frozen):
    r""""""

    def __init__(self, func, base=np.e):
        self._f = func
        self._b = base
        self._freeze()

    def __call__(self, t, x, y, z):
        r"""Compute log_{base} func(t, x, y, z)"""
        f = self._f(t, x, y, z)
        if self._b == np.e:
            return np.log(f)
        else:
            raise NotImplementedError()
