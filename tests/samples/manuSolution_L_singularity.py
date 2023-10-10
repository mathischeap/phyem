# -*- coding: utf-8 -*-
r"""
"""
from numpy import sin, pi, cos
from tools.frozen import Frozen
from tools.functions.space._2d.Cartesian_polar_coordinates_switcher import CartPolSwitcher
from tools.functions.time_space._2d.wrappers.scalar import T2dScalar


class ManufacturedSolutionLSingularity(Frozen):
    """"""

    def __init__(self):
        """"""
        self._freeze()

    @staticmethod
    def phi(t, x, y):
        """"""
        r, theta = CartPolSwitcher.cart2pol(x, y)
        return r ** (2/3) * sin((2/3) * (theta + pi/2)) + 0 * t
        # return sin(pi*x)*cos(pi*y)

    @property
    def potential(self):
        """a"""
        return T2dScalar(self.phi)
