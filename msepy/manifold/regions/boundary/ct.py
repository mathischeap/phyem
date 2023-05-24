# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 4:29 PM on 5/24/2023
"""
from numpy import zeros_like, ones_like
from tools.frozen import Frozen


class MsePyBoundaryRegionCoordinateTransformation(Frozen):
    """"""

    def __init__(self, boundary_region):
        """"""
        self._region = boundary_region
        self._base_region = boundary_region._base_region
        self._axis_and_side = (
            boundary_region._side_index // 2,  # axis,
            boundary_region._side_index % 2   # - side (0) or + side (1)
        )
        self._freeze()

    def mapping(self, *rst):
        """"""
        boundary_rst = list()
        m, n = self._axis_and_side
        for i, _ in enumerate(rst):
            if i == m:
                if n == 0:
                    _ = zeros_like(_)
                elif n == 1:
                    _ = ones_like(_)
            else:
                pass

            boundary_rst.append(_)

        return self._base_region._ct.mapping(*boundary_rst)
