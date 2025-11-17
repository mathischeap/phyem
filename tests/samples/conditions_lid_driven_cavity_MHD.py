# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.frozen import Frozen
from phyem.tools.gen_piece_wise import genpiecewise
from phyem.tools.functions.time_space._2d.wrappers.vector import T2dVector
from phyem.tools.functions.time_space._2d.wrappers.scalar import T2dScalar


class ConditionsLidDrivenCavity_2dMHD_1(Frozen):
    """
    2D MHD lid driven cavity initial condition case 1.


    B.C.:

    velocity lid is:  top wall (y=1)
    B.n = 1 on top and bottom walls (y=1, y=0).
    B.n = 0 on left and right walls.
    Exn = 0 on all boundaries.

    Initial conditions:
    u = (0, 0)
    B = (0, 1)

    """

    def __init__(self, lid_speed=-1):
        self._lid_speed = lid_speed
        self._freeze()

    @property
    def velocity_initial_condition(self):
        """"""
        return T2dVector(self._0_, self._0_)

    @property
    def vorticity_initial_condition(self):
        return T2dScalar(self._0_)

    @property
    def velocity_boundary_condition_tangential(self):
        """"""
        return T2dVector(self._tangential_speed, self._0_)

    @property
    def B_initial_condition(self):
        """"""
        return T2dVector(self._0_, self._1_)

    @property
    def j_initial_condition(self):
        """electric current density"""
        return self.B_initial_condition.rot

    @staticmethod
    def _0_(t, x, y):
        """"""
        return 0 * x + 0 * y + 0 * t

    # noinspection PyUnusedLocal
    @staticmethod
    def _1_(t, x, y):
        """"""
        return np.ones_like(x)

    def _tangential_speed(self, t, x, y):
        """"""
        return genpiecewise([t, x, y], [y <= 0.5, y > 0.5], [self._0_, self.__lid_speed__])

    # noinspection PyUnusedLocal
    def __lid_speed__(self, t, x, y):
        return self._lid_speed * np.ones_like(x)
