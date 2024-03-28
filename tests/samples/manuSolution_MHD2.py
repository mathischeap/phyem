# -*- coding: utf-8 -*-
"""
Manufactured solutions for 2d MHD.

"""
import sys
if './' not in sys.path:
    sys.path.append('./')

from numpy import sin, cos, pi, exp
from tools.frozen import Frozen

from tools.functions.time_space._2d.wrappers.scalar import T2dScalar


def _phi(t, x, y):
    """"""
    return (2 * sin(y) - 2 * cos(x)) * exp(-t)


def _A(t, x, y):
    """"""
    return (cos(2*y) - 2 * cos(x)) * exp(-t)


def _P(t, x, y):
    return sin(x) * sin(y) * exp(-t)


class ManufacturedSolutionMHD2Ideal1(Frozen):
    """"""

    def __init__(self, c=1, Rm=100, Rf=100):
        """

        Parameters
        ----------
        c :
            :math:`Al^{-2}`

        """
        self._c = c
        self._Rm = Rm
        self._Rf = Rf
        self._streaming = T2dScalar(_phi)
        self._potential = T2dScalar(_A)
        self._P = T2dScalar(_P)
        self._freeze()

    @property
    def c(self):
        return self._c

    @property
    def Rm(self):
        return self._Rm

    @property
    def Rf(self):
        return self._Rf

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
        return (1 / self.Rm) * self.j - self.u.cross_product(self.B)

    @property
    def f(self):
        """body force."""
        return (
            self.u.time_derivative + self.omega.cross_product(self.u) + (1 / self.Rf) * self.omega.curl
            - self._c * (self.j.cross_product(self.B)) + self.P.gradient
        )

    @property
    def g(self):
        """magnetic source"""
        return self.B.time_derivative + self.E.curl


if __name__ == '__main__':
    # python tests/samples/manuSolution_MHD2.py
    ic = ManufacturedSolutionMHD2Ideal1()
    ic.j.visualize([0, 2*pi], 1)
