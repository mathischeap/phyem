# -*- coding: utf-8 -*-
"""
Manufactured solutions for 2d MHD.

"""
import sys

if './' not in sys.path:
    sys.path.append('./')

from numpy import sin, cos, exp
from tools.frozen import Frozen

from tools.functions.time_space._3d.wrappers.scalar import T3dScalar
from tools.functions.time_space._3d.wrappers.vector import T3dVector


def ux(t, x, y, z):
    """"""
    # return (sin(x) + cos(y) - (3/2) * sin(z)) * exp(t)
    return cos(x) * sin(y) * sin(z) * exp(t)


def uy(t, x, y, z):
    """"""
    # return ((-cos(x) + sin(z)) * y + 2 * cos(z)) * exp(t)
    return sin(x) * cos(y) * sin(z) * exp(t)


def uz(t, x, y, z):
    """"""
    # return (2 * sin(x) * sin(y) + cos(z)) * exp(t)
    return -2 * sin(x) * sin(y) * cos(z) * exp(t)


def _P_function(t, x, y, z):
    return cos(x) * cos(y) * cos(z) * exp(-t)


# ---------------- Ez -------------------------------------
def Ex(t, x, y, z):
    """"""
    return cos(x) * sin(y) * sin(z / 2) * exp(t)


def dEx_dy(t, x, y, z):
    """"""
    return cos(x) * cos(y) * sin(z / 2) * exp(t)


def dEx_dz(t, x, y, z):
    """"""
    return 0.5 * cos(x) * sin(y) * cos(z / 2) * exp(t)


# ---------------- Ez -------------------------------------
def Ey(t, x, y, z):
    """"""
    return sin(x / 2) * cos(y) * sin(z) * exp(t)


def dEy_dx(t, x, y, z):
    """"""
    return 0.5 * cos(x / 2) * cos(y) * sin(z) * exp(t)


def dEy_dz(t, x, y, z):
    """"""
    return sin(x / 2) * cos(y) * cos(z) * exp(t)


# ---------------- Ez -------------------------------------
def Ez(t, x, y, z):
    """"""
    return - sin(x) * sin(y) * cos(z) * exp(t)


def dEz_dx(t, x, y, z):
    """"""
    return - cos(x) * sin(y) * cos(z) * exp(t)


def dEz_dy(t, x, y, z):
    """"""
    return - sin(x) * cos(y) * cos(z) * exp(t)


# ================================================================
def curlE_x(t, x, y, z):
    return dEz_dy(t, x, y, z) - dEy_dz(t, x, y, z)


def curlE_y(t, x, y, z):
    return dEx_dz(t, x, y, z) - dEz_dx(t, x, y, z)


def curlE_z(t, x, y, z):
    return dEy_dx(t, x, y, z) - dEx_dy(t, x, y, z)


def Bx(t, x, y, z):
    """"""
    return - curlE_x(t, x, y, z) + curlE_x(0, x, y, z)


def By(t, x, y, z):
    """"""
    return - curlE_y(t, x, y, z) + curlE_y(0, x, y, z)


def Bz(t, x, y, z):
    """"""
    return - curlE_z(t, x, y, z) + curlE_z(0, x, y, z)


class ManufacturedSolutionMHD3_0(Frozen):
    """This manufactured solution is used in the decoupled MHD paper."""

    def __init__(self, c=1, Rm=1, Rf=1):
        """

        Parameters
        ----------
        c :
            Coupling number.
        Rm :
        Rf :

        """
        self._c = c
        self._Rm = Rm
        self._Rf = Rf
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
        return T3dVector(ux, uy, uz)

    @property
    def P(self):
        return T3dScalar(_P_function)

    @property
    def omega(self):
        """vorticity"""
        return self.u.curl

    @property
    def E(self):
        """Electronic field."""
        return T3dVector(Ex, Ey, Ez)

    @property
    def B(self):
        """Electronic field."""
        return T3dVector(Bx, By, Bz)

    @property
    def divB(self):
        return self.B.divergence

    @property
    def j(self):
        """electric current density"""
        return self.B.curl

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

    @property
    def g(self):
        return T3dScalar(0)
