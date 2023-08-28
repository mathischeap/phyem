# -*- coding: utf-8 -*-
"""
Manufactured solutions for 2d NS.

"""
import sys

import numpy as np

if './' not in sys.path:
    sys.path.append('./')

from numpy import sin, cos, pi, exp
from tools.frozen import Frozen

from tools.functions.time_space._2d.wrappers.scalar import T2dScalar
from tools.functions.time_space._2d.wrappers.vector import T2dVector


class ManufacturedSolutionNS2TGV(Frozen):
    """"""

    def __init__(self, Re=100):
        """

        Parameters
        ----------
        Re : int, float
        """
        self._Re = Re
        self._velocity = T2dVector(self._u, self._v)
        self._pressure = T2dScalar(self._p)
        self._freeze()

    @property
    def Re(self):
        """Re number."""
        return self._Re

    def _u(self, t, x, y):
        """"""
        return - sin(pi*x) * cos(pi*y) * exp(-2*pi**2 * t / self.Re)

    def _v(self, t, x, y):
        """"""
        return cos(pi*x) * sin(pi*y) * exp(-2*pi**2 * t / self.Re)

    def _p(self, t, x, y):
        """"""
        return 0.25 * (cos(2*pi*x) + cos(2*pi*y)) * exp(-4*pi**2 * t / self.Re)

    @property
    def u(self):
        """fluid velocity field"""
        return self._velocity

    @property
    def omega(self):
        return self.u.rot

    @property
    def p(self):
        """static pressure"""
        return self._pressure

    @property
    def P(self):
        """total pressure"""
        return self.p + 0.5 * (self.u.dot(self.u))

    @property
    def div_u(self):
        return self.u.divergence

    @property
    def f(self):
        """body force: it will be a zero vector."""
        return (self.u.time_derivative + self.omega.cross_product(self.u) +
                (1/self.Re) * self.omega.curl + self.P.gradient)


if __name__ == '__main__':
    # python tests/samples/manuSolution_NS2.py
    ic = ManufacturedSolutionNS2TGV(Re=np.inf)
    ic.omega.visualize([0, 2], 0)
