# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from tools.dds.region_wise_structured import DDSRegionWiseStructured


class MsePyRootFormNumeric(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._freeze()

    def rot(self, *grid_xi_et_sg, time=None):
        """"""
        if time is None:
            time = self._f.cochain.newest
        else:
            pass
        indicator = self._f.space.abstract.indicator
        if indicator == 'bundle':
            space = self._f.space.abstract
            m, n, k = space.m, space.n, space.k
            if m == n == 2 and k == 0:
                df = self._f.coboundary[time]()

                xy, tensor = df.reconstruct(time, *grid_xi_et_sg)

                x, y = xy

                pv_px = tensor[1][0]
                pu_py = tensor[0][1]

                rot_data = pv_px - pu_py

                x, y, rot_data = self._f.mesh._regionwsie_stack(x, y, rot_data)

                return DDSRegionWiseStructured([x, y], [rot_data, ])

            else:
                raise Exception(f"form of space {space} cannot perform curl.")
