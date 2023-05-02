# -*- coding: utf-8 -*-
from abc import ABC
from tools.functions.space._2d.constant import CFG


class Opposite(ABC):
    """
    Equal to ``ScalingFunc(-1)(func)``.

    .. doctest::

        >>> new_func = Opposite(CFG(5)())()
        >>> new_func(0, 0)
        -5.0
    """

    def __init__(self, func):
        self._func_ = func

    def _opposite_func_(self, x, y):
        return -self._func_(x, y)

    def __call__(self):
        return self._opposite_func_


if __name__ == "__main__":
    import doctest
    doctest.testmod()
