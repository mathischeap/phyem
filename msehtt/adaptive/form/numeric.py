# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.frozen import Frozen
from phyem.src.config import RANK, MASTER_RANK


class MseHtt_Adaptive_TopForm_Numeric(Frozen):
    r""""""

    def __init__(self, tf):
        r""""""
        self._tf = tf
        self._freeze()

    @property
    def abs(self):
        r"""Only return in the master rank, return None on other ranks."""
        dtype, itp = self._tf[None].numeric.interpolate(component_wise=False, rankwise=False)
        if dtype == '2d-scalar':
            if RANK == MASTER_RANK:
                return ___ABS_WRAPPER___(itp)
            else:
                return None
        else:
            raise NotImplementedError(f"{self.__class__.__name__} abs dtype={dtype} not implemented")


class ___ABS_WRAPPER___(Frozen):
    r""""""

    def __init__(self, func):
        r""""""
        self._func = func

    def __call__(self, *args, **kwargs):
        r""""""
        return np.abs(self._func(*args, **kwargs))
