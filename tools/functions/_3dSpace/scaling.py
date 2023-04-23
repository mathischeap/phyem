# -*- coding: utf-8 -*-
from abc import ABC


class ScalingFunc(ABC):
    """Scaling a function: new_func = C*func."""
    def __init__(self, C):
        self._C_ = C

    def _scaled_func_(self, x, y, z):
        return self._C_ * self._func_(x, y, z)

    def __call__(self, func):
        self._func_ = func
        return self._scaled_func_
