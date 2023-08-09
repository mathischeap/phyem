# -*- coding: utf-8 -*-
"""
Manufactured solutions for 2d MHD.

"""
import sys
if './' not in sys.path:
    sys.path.append('./')

from numpy import sin, cos, pi
from tools.frozen import Frozen

from tools.functions.time_space._2d.wrappers.scalar import T2dScalar


def _phi(t, x, y):
    """"""
    return 2 * sin(y) - 2 * cos(x) + sin(t)


def _A(t, x, y):
    """"""
    return cos(2*y) - 2 * cos(x) + cos(t)


def _P(t, x, y):
    return sin(x) * sin(y) + sin(t)


class ManufacturedSolutionMHD2Ideal1(Frozen):
    """"""

    def __init__(self, c=1):
        """

        Parameters
        ----------
        c :
            :math:`Al^{-2}`

        """
        self._c = c
        self._streaming = T2dScalar(_phi)
        self._potential = T2dScalar(_A)
        self._P = T2dScalar(_P)
        self._freeze()

    @property
    def u(self):
        """fluid velocity field"""
        return self._streaming.curl

    @property
    def B(self):
        """magnetic flux density"""
        return self._potential.curl

    @property
    def P(self):
        return self._P

    @property
    def omega(self):
        """vorticity"""
        return self.u.rot

    @property
    def j(self):
        """electric current density"""
        return self.B.rot

    @property
    def E(self):
        """electric field strength"""
        return - self.u.cross_product(self.B)

    @property
    def f(self):
        """body force."""
        return (
            self.u.time_derivative + self.omega.cross_product(self.u) -
            self._c * (self.j.cross_product(self.B)) + self.P.gradient
        )

    @property
    def g(self):
        """magnetic source"""
        return self.B.time_derivative + self.E.curl


if __name__ == '__main__':
    # python tests/samples/manuSolution_MHD2.py
    ic = ManufacturedSolutionMHD2Ideal1()
    ic.B.visualize([0, 2*pi], 0)
