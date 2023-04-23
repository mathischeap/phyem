# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 1:53 PM on 4/17/2023
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
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
        """"""
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
        ndim = manifold.ndim
        linspace = np.linspace(0, 1, samples)
        zeros = np.zeros(samples)
        ones = np.ones(samples)
        Lines = dict()
        for i in manifold.regions:  # region #i
            ct = manifold.ct
            if ndim == 1:   # manifold ndim == 1
                rst = np.array([0, 1])
                line = ct.mapping(rst, regions=i)[i]
                Lines[i] = (line, )  # for 1-d region, no matter what esd is, it contains one line.

            elif ndim == 2:  # manifold ndim == 2
                # there will be four lines
                line_upper = ct.mapping(zeros, linspace, regions=i)[i]
                line_down = ct.mapping(ones, linspace, regions=i)[i]
                line_left = ct.mapping(linspace, zeros, regions=i)[i]
                line_right = ct.mapping(linspace, ones, regions=i)[i]
                Lines[i] = (line_upper, line_down, line_left, line_right)

            elif ndim == 3:  # manifold ndim == 3
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

                Lines[i] = (
                    line_dx_WB, line_dx_EB, line_dx_WF, line_dx_EF,
                    line_dy_NB, line_dy_SB, line_dy_NF, line_dy_SF,
                    line_dz_NW, line_dz_SW, line_dz_NE, line_dz_SE,
                )

        return Lines


if __name__ == '__main__':
    # python msepy/manifold/visualize/main.py
    import __init__ as ph
    space_dim = 3
    ph.config.set_embedding_space_dim(space_dim)

    manifold = ph.manifold(space_dim)
    mesh = ph.mesh(manifold)

    msepy, obj = ph.fem.apply('msepy', locals())

    mnf = obj['manifold']
    msh = obj['mesh']

    # msepy.config(mnf)('crazy', c=0., periodic=True, bounds=[[0, 2] for _ in range(space_dim)])
    msepy.config(mnf)('backward_step')
    msepy.config(msh)([3 for _ in range(space_dim)])

    mnf.visualize()
