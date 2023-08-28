# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from tools.dds.region_wise_structured import DDSRegionWiseStructured


class MsePyRootFormNumeric(Frozen):
    """"""

    def __init__(self, rf, time):
        """"""
        self._f = rf
        self._time = time
        self._freeze()

    def rot(self, *grid_xi_et_sg):
        """"""
        time = self._time
        indicator = self._f.space.abstract.indicator
        if indicator == 'bundle':
            space = self._f.space.abstract
            m, n, k = space.m, space.n, space.k
            if m == n == 2 and k == 0:
                df = self._f.coboundary[time]

                xy, tensor = df.reconstruct(*grid_xi_et_sg)

                x, y = xy

                pv_px = tensor[1][0]
                pu_py = tensor[0][1]

                rot_data = pv_px - pu_py

                x, y, rot_data = self._f.mesh._regionwsie_stack(x, y, rot_data)

                return DDSRegionWiseStructured([x, y], [rot_data, ])

            else:
                raise Exception(f"form of space {space} cannot perform curl.")
        else:
            raise NotImplementedError()

    def divergence(self, *grid_xi_et_sg, magnitude=False):
        """"""
        time = self._time
        indicator = self._f.space.abstract.indicator
        if indicator == 'bundle':
            space = self._f.space.abstract
            m, n, k = space.m, space.n, space.k
            if m == n == 2 and k == 0:
                df = self._f.coboundary[time]

                xy, tensor = df.reconstruct(*grid_xi_et_sg)

                x, y = xy

                pu_px = tensor[0][0]
                pv_py = tensor[1][1]

                div_data = pu_px + pv_py
                if magnitude:
                    div_data = np.abs(div_data)
                    div_data[div_data < 1e-16] = 1e-16
                    div_data = np.log10(div_data)
                else:
                    pass

                x, y, div_data = self._f.mesh._regionwsie_stack(x, y, div_data)

                return DDSRegionWiseStructured([x, y], [div_data, ])

            else:
                raise Exception(f"form of space {space} cannot perform curl.")
        else:
            raise NotImplementedError()
