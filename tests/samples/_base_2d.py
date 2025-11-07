# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


# noinspection PyUnusedLocal
def _0_(t, x, y):
    r""""""
    return 0 * x


# noinspection PyUnusedLocal
def _1_(t, x, y):
    r""""""
    return 0 * x + 1


# noinspection PyUnusedLocal
def _half_(t, x, y):
    r""""""
    return 0 * x + 0.5


# noinspection PyUnusedLocal
def _m1_(t, x, y):
    r""""""
    return 0 * x - 1


from tools.functions.time_space._2d.wrappers.vector import T2dVector
from tools.functions.time_space._2d.wrappers.scalar import T2dScalar


class _TwoDimensionalConditionBase(Frozen):
    r""""""

    @property
    def zero_scalar(self):
        return T2dScalar(_0_, steady=True)

    @property
    def zero_vector(self):
        return T2dVector(_0_, _0_, steady=True)

    @property
    def one_scalar(self):
        return T2dScalar(_1_, steady=True)

    @property
    def half_scalar(self):
        return T2dScalar(_half_, steady=True)

    @property
    def minus_one_scalar(self):
        return T2dScalar(_m1_, steady=True)
