# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.src.config import RANK, MASTER_RANK


def rectangle(bounds=([0, 1], [0, 1]), periodic=False):
    r"""Mainly for test purpose."""
    assert RANK == MASTER_RANK, f"only initialize rectangle mesh in the master rank"

    if periodic:
        raise NotImplementedError()
    else:
        pass

    low_x, upp_x = bounds[0]
    low_y, upp_y = bounds[1]

    assert low_x < upp_x and low_y < upp_y, f"bounds = {bounds} wrong."

    REGIONS = {
        0: _MAP_(low_x, upp_x, low_y, upp_y)
    }

    region_map = None        # the config method will parse the region map.
    periodic_setting = None

    return REGIONS, region_map, periodic_setting


class _MAP_(Frozen):
    r""""""
    def __init__(self, xl, xu, yl, yu):
        r""""""
        self._xl_ = xl
        self._yl_ = yl
        self._X_ = xu - xl
        self._Y_ = yu - yl
        self._freeze()

    @property
    def ndim(self):
        r"""This is a 2d region."""
        return 2

    @property
    def etype(self):
        r"""The element made in this region can only be of this type."""
        return 9

    def mapping(self, r, s):
        r""""""
        x = r * self._X_ + self._xl_
        y = s * self._Y_ + self._yl_
        return x, y

    # noinspection PyUnusedLocal
    def Jacobian_matrix(self, r, s):
        r""""""
        return (
            (self._X_, 0),
            (0, self._Y_)
        )
