# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msepy.manifold.visualize.matplot import MsePyManifoldVisualizeMatplot
import numpy as np


class MsePyManifoldVisualize(Frozen):
    """"""

    def __init__(self, manifold):
        """"""
        self._manifold = manifold
        self._matplot = None
        self._freeze()

    def __call__(self, *args, **kwargs):
        """"""
        return self.matplot(*args, **kwargs)

    @property
    def matplot(self):
        if self._matplot is None:
            self._matplot = MsePyManifoldVisualizeMatplot(self._manifold)
        return self._matplot

    def _generate_manifold_grid_data(
            self,
            refining_factor=1,
    ):
        """Make the region edge lines."""
        if refining_factor <= 0.1:
            refining_factor = 0.1
        samples = 100 * refining_factor
        if samples <= 10:
            samples = 10
        elif samples > 1000:
            samples = 1000
        else:
            pass
        manifold = self._manifold
        m = manifold.m
        n = manifold.n
        linspace = np.linspace(0, 1, samples)
        zeros = np.zeros(samples)
        ones = np.ones(samples)

        Region_lines = dict()  # keys regions, values: list of lines

        if m == n:

            for i in manifold.regions:  # region #i
                ct = manifold.ct
                if n == 1:   # manifold ndim == 1
                    rst = np.array([0, 1])
                    line = ct.mapping(rst, regions=i)[i]
                    Region_lines[i] = (line,)  # for 1-d region, no matter what esd is, it contains one line.
                elif n == 2:  # manifold ndim == 2
                    # there will be four lines
                    line_upper = ct.mapping(zeros, linspace, regions=i)[i]
                    line_down = ct.mapping(ones, linspace, regions=i)[i]
                    line_left = ct.mapping(linspace, zeros, regions=i)[i]
                    line_right = ct.mapping(linspace, ones, regions=i)[i]
                    Region_lines[i] = (line_upper, line_down, line_left, line_right)

                elif n == 3:  # manifold ndim == 3
                    # there will be 12 lines
                    line_dx_WB = ct.mapping(linspace, zeros, zeros, regions=i)[i]
                    line_dx_EB = ct.mapping(linspace, ones, zeros, regions=i)[i]
                    line_dx_WF = ct.mapping(linspace, zeros, ones, regions=i)[i]
                    line_dx_EF = ct.mapping(linspace, ones, ones, regions=i)[i]

                    line_dy_NB = ct.mapping(zeros, linspace, zeros, regions=i)[i]
                    line_dy_SB = ct.mapping(ones, linspace, zeros, regions=i)[i]
                    line_dy_NF = ct.mapping(zeros, linspace, ones, regions=i)[i]
                    line_dy_SF = ct.mapping(ones, linspace, ones, regions=i)[i]

                    line_dz_NW = ct.mapping(zeros, zeros, linspace, regions=i)[i]
                    line_dz_SW = ct.mapping(ones, zeros, linspace, regions=i)[i]
                    line_dz_NE = ct.mapping(zeros, ones, linspace, regions=i)[i]
                    line_dz_SE = ct.mapping(ones, ones, linspace, regions=i)[i]

                    Region_lines[i] = (
                        line_dx_WB, line_dx_EB, line_dx_WF, line_dx_EF,
                        line_dy_NB, line_dy_SB, line_dy_NF, line_dy_SF,
                        line_dz_NW, line_dz_SW, line_dz_NE, line_dz_SE,
                    )
                else:
                    raise Exception("should never reach this spot.")

        elif m == 1 and n == 0:

            region_map_type = manifold.regions._map_type

            if region_map_type == 1:  # region-boundary-type regions.

                for i in manifold.regions:  # region #i
                    ct = manifold.regions[i]._ct
                    node = ct.mapping(np.zeros(1))[0]
                    Region_lines[i] = (node, )

            else:
                raise NotImplementedError(
                    f"from m={m}, n={n}, cannot get lines for region map type = {region_map_type}"
                )

        elif m == 2 and n == 1:

            region_map_type = manifold.regions._map_type

            if region_map_type == 1:  # region-boundary-type regions.

                for i in manifold.regions:  # region #i
                    ct = manifold.regions[i]._ct
                    line = ct.mapping(linspace)
                    Region_lines[i] = (line, )

            else:
                raise NotImplementedError(
                    f"from m={m}, n={n}, cannot get lines for region map type = {region_map_type}"
                )

        elif m == 3 and n == 2:

            region_map_type = manifold.regions._map_type

            if region_map_type == 1:  # region-boundary-type regions.

                for i in manifold.regions:  # region #i
                    ct = manifold.regions[i]._ct
                    _2d_line_space = np.meshgrid(linspace, linspace, indexing='ij')
                    _2d_face_grid_data = ct.mapping(*_2d_line_space)
                    Region_lines[i] = _2d_face_grid_data

            else:
                raise NotImplementedError(
                    f"from m={m}, n={n}, cannot get lines for region map type = {region_map_type}"
                )

        else:
            raise NotImplementedError()

        return Region_lines
