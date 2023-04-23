# -*- coding: utf-8 -*-
from abc import ABC


class Opposite(ABC):
    """Equal to ``ScalingFunc(-1, func)()``."""
    def __init__(self, func):
        """ """
        self._func_ = func

    def _opposite_func_(self, x, y, z):
        return -self._func_(x, y, z)

    def __call__(self):
        return self._opposite_func_
