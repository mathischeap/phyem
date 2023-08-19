# -*- coding: utf-8 -*-
r"""
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
        num_rst = len(rst)

        if num_rst == self._region.m:
            boundary_rst = list()
            m, n = self._axis_and_side
            for i, _ in enumerate(rst):
                if i == m:
                    if n == 0:
                        _ = zeros_like(_)
                    elif n == 1:
                        _ = ones_like(_)
                    else:
                        raise Exception
                else:
                    pass

                boundary_rst.append(_)

        elif num_rst == self._region.m - 1:

            boundary_rst = list(rst)
            m, n = self._axis_and_side

            if n == 0:
                _ = zeros_like(boundary_rst[0])
            elif n == 1:
                _ = ones_like(boundary_rst[0])
            else:
                raise Exception

            boundary_rst.insert(m, _)

        else:
            raise NotImplementedError()

        return self._base_region._ct.mapping(*boundary_rst)
