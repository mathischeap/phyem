# -*- coding: utf-8 -*-
from abc import ABC
from tools.functions._2dSpace.constant import CFG


class ScalingFunc(ABC):
    """
    Scaling a function: new_func = C*func.

    To scale a function, saying `func`, with constant C, do for example:

    .. doctest::

        >>> new_func = ScalingFunc(3)(CFG(5)())
        >>> new_func(0, 0)
        15.0

    ``newfunc`` is the scaled function which alway return 15.
    """

    def __init__(self, C):
        self._C_ = C

    def _scaled_func_(self, x, y):
        return self._C_ * self._func_(x, y)

    def __call__(self, func):
        self._func_ = func
        return self._scaled_func_


if __name__ == "__main__":
    import doctest
    doctest.testmod()
