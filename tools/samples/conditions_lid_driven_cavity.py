# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.frozen import Frozen
from phyem.tools.gen_piece_wise import genpiecewise
from phyem.tools.functions.time_space._2d.wrappers.vector import T2dVector
from phyem.tools.functions.time_space._3d.wrappers.vector import T3dVector


class ConditionsLidDrivenCavity2(Frozen):
    """"""

    def __init__(self, lid_speed=-1):
        self._lid_speed = lid_speed
        self._freeze()

    @property
    def velocity_initial_condition(self):
        """"""
        return T2dVector(self._0_, self._0_)

    @property
    def vorticity_initial_condition(self):
        return self.velocity_initial_condition.rot

    @property
    def velocity_boundary_condition_tangential(self):
        """"""
        return T2dVector(self._tangential_speed, self._0_)

    @staticmethod
    def _0_(t, x, y):
        """"""
        return 0 * x + 0 * y + 0 * t

    def _tangential_speed(self, t, x, y):
        """"""
        return genpiecewise([t, x, y], [y <= 0.5, y > 0.5], [self._0_, self.__lid_speed__])

    def __lid_speed__(self, t, x, y):
        return self._lid_speed * np.ones_like(x) + 0 * x + 0 * y + 0 * t


class ConditionsLidDrivenCavity3(Frozen):
    """"""

    def __init__(self, lid_speed=-1):
        self._lid_speed = lid_speed
        self._freeze()

    @property
    def velocity_initial_condition(self):
        """"""
        return T3dVector(self._0_, self._0_, self._0_)

    @property
    def vorticity_initial_condition(self):
        return self.velocity_initial_condition.curl

    @property
    def velocity_boundary_condition_tangential(self):
        """"""
        return T3dVector(self._0_, self._0_, self._tangential_speed)

    @staticmethod
    def _0_(t, x, y, z):
        """"""
        return 0 * x + 0 * y + 0 * z + 0 * t

    def _tangential_speed(self, t, x, y, z):
        """"""
        return genpiecewise([t, x, y, z], [y <= 0.5, y > 0.5], [self._0_, self.__lid_speed__])

    def __lid_speed__(self, t, x, y, z):
        return self._lid_speed * np.ones_like(x) + 0 * x * y * z + 0 * t
