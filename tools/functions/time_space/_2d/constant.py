# -*- coding: utf-8 -*-
from abc import ABC
import numpy as np


_cache_CFGt_m2n2_ = {}


def cfg_t(C):
    r"""A wrapper of CFGt."""

    if C in _cache_CFGt_m2n2_:
        return _cache_CFGt_m2n2_[C]
    else:
        cfg = CFGt(C)()
        _cache_CFGt_m2n2_[C] = cfg
        return cfg


class CFGt(ABC):
    """
    Constant function generator: ``CF = CFG(5)()``.

    .. doctest::

        >>> cf = CFGt(10.5)()
        >>> float(cf(1, 1,1))
        10.5
    """

    def __init__(self, C):
        """ """
        self._C_ = C

    def _constant_func_(self, t, x, y):
        assert np.shape(x) == np.shape(y)
        return self._C_ + np.zeros(np.shape(x)) + 0 * t

    def __call__(self):
        return self._constant_func_


if __name__ == "__main__":
    import doctest
    doctest.testmod()
