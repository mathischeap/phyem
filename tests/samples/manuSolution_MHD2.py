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
from tools.functions.time_space._2d.wrappers.vector import T2dVector


def _phi(t, x, y):
    """"""
    return (2 * sin(y) - 2 * cos(x)) * exp(-t)


def _P_function(t, x, y):
    return sin(x) * sin(y) * exp(-t)


def _E_function(t, x, y):
    """"""
    return sin(x) * sin(y) * exp(t)


def _curl_E_x(t, x, y):
    return sin(x) * cos(y) * exp(t)


def _curl_E_y(t, x, y):
    return - cos(x) * sin(y) * exp(t)


def _int_0_t_curl_E_x(t, x, y):
    """"""
    return sin(x) * cos(y) * (exp(t) - 1)


def _int_0_t_curl_E_y(t, x, y):
    """"""
    return - cos(x) * sin(y) * (exp(t) - 1)


def _Bx(t, x, y):
    return sin(x) * cos(y) - _int_0_t_curl_E_x(t, x, y)


def _By(t, x, y):
    return - cos(x) * sin(y) - _int_0_t_curl_E_y(t, x, y)


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
        self._streamfunction = T2dScalar(_phi)
        self._P = T2dScalar(_P_function)
        self._E = T2dScalar(_E_function)
        self._B = T2dVector(_Bx, _By)
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
    def streamfunction(self):
        return self._streamfunction

    @property
    def u(self):
        """fluid velocity field"""
        return self._streamfunction.curl

    @property
    def P(self):
        return self._P

    @property
    def omega(self):
        """vorticity"""
        return self.u.rot

    @property
    def E(self):
        """Electronic field."""
        return self._E

    @property
    def B(self):
        return self._B

    @property
    def divB(self):
        return self.B.divergence

    @property
    def j(self):
        """electric current density"""
        return self.B.rot

    @property
    def m(self):
        """electric source.

        (1/Rm)j - E - u x B = m
        """
        # noinspection PyTypeChecker
        return (1 / self.Rm) * self.j - self.E - self.u.cross_product(self.B)

    @property
    def f(self):
        """body force."""
        return (
            self.u.time_derivative + self.omega.cross_product(self.u) + (1 / self.Rf) * self.omega.curl
            - self._c * (self.j.cross_product(self.B)) + self.P.gradient
        )


if __name__ == '__main__':
    # python tests/samples/manuSolution_MHD2.py
    ic = ManufacturedSolutionMHD2Ideal1()
    ic.E.visualize([0, 2*pi], 0.151231)
    # ic.f.visualize([0, 2*pi], 0.151231)
