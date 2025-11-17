# -*- coding: utf-8 -*-
"""
Manufactured solutions for 3d NS.

"""
from numpy import sin, cos, pi

from phyem.tools.frozen import Frozen
from phyem.tools.functions.time_space._3d.wrappers.vector import T3dVector


# noinspection PyUnusedLocal
class ManufacturedSolutionNS3Conservation1(Frozen):
    """The conservation test used in the JCP 2022 Dual-NS paper, section 5.1.1."""

    def __init__(self, Re=100):
        """

        Parameters
        ----------
        Re : int, float
        """
        self._Re = Re
        self._velocity = T3dVector(self._u, self._v, self._w)
        self._freeze()

    @property
    def Re(self):
        """Re number."""
        return self._Re

    @staticmethod
    def _u(t, x, y, z):
        """"""
        return cos(2 * pi * z)

    @staticmethod
    def _v(t, x, y, z):
        """"""
        return sin(2 * pi * z)

    @staticmethod
    def _w(t, x, y, z):
        """"""
        return sin(2 * pi * x)

    @property
    def u(self):
        """fluid velocity field"""
        return self._velocity

    @property
    def div_u(self):
        return self.u.divergence

    @property
    def omega(self):
        return self.u.curl
