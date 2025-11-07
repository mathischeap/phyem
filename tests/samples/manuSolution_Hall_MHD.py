# -*- coding: utf-8 -*-
"""
Manufactured solutions for 3D Hall-MHD.

"""
import sys

if './' not in sys.path:
    sys.path.append('./')

from numpy import sin, cos, exp, pi, zeros_like
from tools.frozen import Frozen

from tools.functions.time_space._3d.wrappers.scalar import T3dScalar
from tools.functions.time_space._3d.wrappers.vector import T3dVector


# ================ MANU 0: [0, 2pi]^3, periodic domain =======================================

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


# ---------------- Ex -------------------------------------
def Ex(t, x, y, z):
    """"""
    return cos(x) * sin(y) * sin(z) * exp(t)


def dEx_dy(t, x, y, z):
    """"""
    return cos(x) * cos(y) * sin(z) * exp(t)


def dEx_dz(t, x, y, z):
    """"""
    return cos(x) * sin(y) * cos(z) * exp(t)


# ---------------- Ey -------------------------------------
def Ey(t, x, y, z):
    """"""
    return sin(x) * cos(y) * sin(z) * exp(t)


def dEy_dx(t, x, y, z):
    """"""
    return cos(x) * cos(y) * sin(z) * exp(t)


def dEy_dz(t, x, y, z):
    """"""
    return sin(x) * cos(y) * cos(z) * exp(t)


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


# ------------ B -------------------------------------------------------
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


class ManufacturedSolution_Hall_MHD3_0(Frozen):
    """This manufactured solution is for domain [0, 2pi]^3 x (0, t].

    It is periodic in this domain.
    """

    def __init__(self, c=1, Rm=1, Rf=1, eta=1):
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
        self._eta = eta

        self._u = T3dVector(ux, uy, uz)
        self._P = T3dScalar(_P_function)
        self._E = T3dVector(Ex, Ey, Ez)
        self._B = T3dVector(Bx, By, Bz)

        self._m = None
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
    def eta(self):
        return self._eta

    @property
    def u(self):
        return self._u

    @property
    def P(self):
        return self._P

    @property
    def omega(self):
        """vorticity"""
        return self.u.curl

    @property
    def E(self):
        return self._E

    @property
    def B(self):
        return self._B

    @property
    def H(self):
        return self._B

    @property
    def divB(self):
        return self.B.divergence

    @property
    def div_u(self):
        return self.u.divergence

    @property
    def j(self):
        """electric current density"""
        return self.B.curl

    @property
    def m(self):
        """electric source.

        (1/Rm)j - E - u x B + eta * j x B = m
        """
        if self._m is None:
            self._m = (
                (1 / self.Rm) * self.j
                - self.E
                - self.u.cross_product(self.B)
                + self.eta * (self.j.cross_product(self.B))
            )
        return self._m

    @property
    def f(self):
        """body force."""
        return (
            self.u.time_derivative + self.omega.cross_product(self.u) + (1 / self.Rf) * self.omega.curl
            - self._c * (self.j.cross_product(self.B)) + self.P.gradient
        )


# ================ MANU 1: [0, 2pi]^3, BC: uxn =0, P=0, Bxn = 0 ===============================================

def ux1(t, x, y, z):
    """"""
    return cos(x) * sin(y) * sin(z) * exp(t)


def uy1(t, x, y, z):
    """"""
    return sin(x) * cos(y) * sin(z) * exp(t)


def uz1(t, x, y, z):
    """"""
    return -2 * sin(x) * sin(y) * cos(z) * exp(t)


def _P_function1(t, x, y, z):
    return sin(x) * sin(y) * sin(z) * exp(-t)


# ---------------- Ex -------------------------------------
# noinspection PyUnusedLocal
def Ex1(t, x, y, z):
    """"""
    return sin(x) * cos(y) * exp(t)


# noinspection PyUnusedLocal
def dEx_dy1(t, x, y, z):
    """"""
    return - sin(x) * sin(y) * exp(t)


# noinspection PyUnusedLocal
def dEx_dz1(t, x, y, z):
    """"""
    return 0 * z


# ---------------- Ey -------------------------------------
# noinspection PyUnusedLocal
def Ey1(t, x, y, z):
    """"""
    return - sin(y) * cos(z) * exp(t)


# noinspection PyUnusedLocal
def dEy_dx1(t, x, y, z):
    """"""
    return 0 * t


# noinspection PyUnusedLocal
def dEy_dz1(t, x, y, z):
    """"""
    return sin(y) * sin(z) * exp(t)


# ---------------- Ez -------------------------------------
# noinspection PyUnusedLocal
def Ez1(t, x, y, z):
    """"""
    return - cos(x) * sin(z) * exp(t)


# noinspection PyUnusedLocal
def dEz_dx1(t, x, y, z):
    """"""
    return sin(x) * sin(z) * exp(t)


# noinspection PyUnusedLocal
def dEz_dy1(t, x, y, z):
    """"""
    return 0 * y


# ---------- B ------------------------------------------------
def curlE_x1(t, x, y, z):
    return dEz_dy1(t, x, y, z) - dEy_dz1(t, x, y, z)


def curlE_y1(t, x, y, z):
    return dEx_dz1(t, x, y, z) - dEz_dx1(t, x, y, z)


def curlE_z1(t, x, y, z):
    return dEy_dx1(t, x, y, z) - dEx_dy1(t, x, y, z)


def Bx1(t, x, y, z):
    """"""
    return - curlE_x1(t, x, y, z) + curlE_x1(0, x, y, z)


def By1(t, x, y, z):
    """"""
    return - curlE_y1(t, x, y, z) + curlE_y1(0, x, y, z)


def Bz1(t, x, y, z):
    """"""
    return - curlE_z1(t, x, y, z) + curlE_z1(0, x, y, z)


class ManufacturedSolution_Hall_MHD3_1(ManufacturedSolution_Hall_MHD3_0):
    r"""This manufactured solution is for domain [0, 2pi]^3 x (0, t].

    It is for vanishing b.c. uxn =0, P=0, Bxn = 0 on whole boundary.
    """

    def __init__(self, c=1, Rm=1, Rf=1, eta=1):
        r"""

        Parameters
        ----------
        c :
            Coupling number.
        Rm :
        Rf :

        """
        super().__init__(c=c, Rm=Rm, Rf=Rf, eta=eta)
        self._melt()

        self._u = T3dVector(ux1, uy1, uz1)
        self._P = T3dScalar(_P_function1)
        self._E = T3dVector(Ex1, Ey1, Ez1)
        self._B = T3dVector(Bx1, By1, Bz1)

        self._freeze()


# ================ MANU 2: [0, 2pi]^3, BC: uxn =0, P=0, Exn = 0 ===============================================


class ManufacturedSolution_Hall_MHD3_2(ManufacturedSolution_Hall_MHD3_0):
    r"""This manufactured solution is for domain [0, 2pi]^3 x (0, t].

    It is for vanishing b.c. uxn =0, P=0, Exn = 0 on whole boundary.
    """
    def __init__(self, c=1, Rm=1, Rf=1, eta=1):
        r"""

        Parameters
        ----------
        c :
            Coupling number.
        Rm :
        Rf :

        """
        super().__init__(c=c, Rm=Rm, Rf=Rf, eta=eta)
        self._melt()

        self._u = T3dVector(ux1, uy1, uz1)
        self._P = T3dScalar(_P_function1)
        self._E = T3dVector(Ex, Ey, Ez)
        self._B = T3dVector(Bx, By, Bz)

        self._freeze()


# ======== MANU 3: [0, 2*pi]^3, P = 0, omega x n = 0, B x n = 0 on whole boundary ==============

# ---------------- Ux -------------------------------------
# noinspection PyUnusedLocal
def Ux1(t, x, y, z):
    """"""
    return sin(x) * cos(y) * exp(t)


# noinspection PyUnusedLocal
def dUx_dy1(t, x, y, z):
    """"""
    return - sin(x) * sin(y) * exp(t)


# noinspection PyUnusedLocal
def dUx_dz1(t, x, y, z):
    """"""
    return 0 * z


# ---------------- Uy -------------------------------------
# noinspection PyUnusedLocal
def Uy1(t, x, y, z):
    """"""
    return - sin(y) * cos(z) * exp(t)


# noinspection PyUnusedLocal
def dUy_dx1(t, x, y, z):
    """"""
    return 0 * t


# noinspection PyUnusedLocal
def dUy_dz1(t, x, y, z):
    """"""
    return sin(y) * sin(z) * exp(t)


# ---------------- Uz -------------------------------------
# noinspection PyUnusedLocal
def Uz1(t, x, y, z):
    """"""
    return - cos(x) * sin(z) * exp(t)


# noinspection PyUnusedLocal
def dUz_dx1(t, x, y, z):
    """"""
    return sin(x) * sin(z) * exp(t)


# noinspection PyUnusedLocal
def dUz_dy1(t, x, y, z):
    """"""
    return 0 * y


# ---------- B ------------------------------------------------
def omega_x(t, x, y, z):
    return dUz_dy1(t, x, y, z) - dUy_dz1(t, x, y, z)


def omega_y(t, x, y, z):
    return dUx_dz1(t, x, y, z) - dUz_dx1(t, x, y, z)


def omega_z(t, x, y, z):
    return dUy_dx1(t, x, y, z) - dUx_dy1(t, x, y, z)


class ManufacturedSolution_Hall_MHD3_3(ManufacturedSolution_Hall_MHD3_0):
    r"""This manufactured solution is for domain [0, 2pi]^3 x (0, t].

    It is for vanishing b.c. omega x n = 0, P = 0, B x n = 0 on whole boundary.
    """
    def __init__(self, c=1, Rm=1, Rf=1, eta=1):
        r"""

        Parameters
        ----------
        c :
            Coupling number.
        Rm :
        Rf :

        """
        super().__init__(c=c, Rm=Rm, Rf=Rf, eta=eta)
        self._melt()

        self._u = T3dVector(Ux1, Uy1, Uz1)
        self._P = T3dScalar(_P_function1)
        self._E = T3dVector(Ex1, Ey1, Ez1)
        self._B = T3dVector(Bx1, By1, Bz1)

        self._omega = T3dVector(omega_x, omega_y, omega_z)

        self._freeze()

    @property
    def omega(self):
        """vorticity"""
        return self._omega


# ======== MANU 4: [0, 1]^3, P = 0, u x n = 0, B x n = 0 on whole boundary for TEMPORAL convergence tests======

def T4_u(t, x, y, z):
    """"""
    return ((2/3)*x**3-x**2) * (y-y**2) * z * (z-1) * exp(t)


def T4_v(t, x, y, z):
    """"""
    return (-x**2 + x) * (0.5*y**2-(1/3)*y**3) * (z**2 - z) * exp(t)


def T4_w(t, x, y, z):
    """"""
    return (-x**2 + x) * (y-y**2) * ((1/3)*z**3 - 0.5*z**2) * exp(t)


def T4_P_func(t, x, y, z):
    return - x**2 * (x - 1) * y**2 * (y - 1) * z**2 * (z - 1) * exp(t)


# ---------------- Ex -------------------------------------
# noinspection PyUnusedLocal
def T4_Ex(t, x, y, z):
    """"""
    return x * (x**2 - 1) * (y**3 - 1.5*y**2) * exp(t)


# noinspection PyUnusedLocal
def T4_Ex_dy(t, x, y, z):
    """"""
    return x * (x**2 - 1) * 3 * (y**2 - y) * exp(t)


# noinspection PyUnusedLocal
def T4_Ex_dz(t, x, y, z):
    """"""
    return 0 * z


# ---------------- Ey -------------------------------------
# noinspection PyUnusedLocal
def T4_Ey(t, x, y, z):
    """"""
    return - y * (y - 1) * ((2/3) * z**3-z**2) * exp(t)


# noinspection PyUnusedLocal
def T4_Ey_dx(t, x, y, z):
    """"""
    return 0 * t


# noinspection PyUnusedLocal
def T4_Ey_dz(t, x, y, z):
    """"""
    return - y * (y - 1) * 2 * (z**2-z) * exp(t)


# ---------------- Ez -------------------------------------
# noinspection PyUnusedLocal
def T4_Ez(t, x, y, z):
    """"""
    return ((1/3) * x**3 - 0.5 * x**2) * z * (z**2 - 1) * exp(t)


# noinspection PyUnusedLocal
def T4_Ez_dx(t, x, y, z):
    """"""
    return (x**2 - x) * z * (z**2 - 1) * exp(t)


# noinspection PyUnusedLocal
def T4_Ez_dy(t, x, y, z):
    """"""
    return 0 * y


# ---------- B ------------------------------------------------
def T4curlE_x1(t, x, y, z):
    return T4_Ez_dy(t, x, y, z) - T4_Ey_dz(t, x, y, z)


def T4curlE_y1(t, x, y, z):
    return T4_Ex_dz(t, x, y, z) - T4_Ez_dx(t, x, y, z)


def T4curlE_z1(t, x, y, z):
    return T4_Ey_dx(t, x, y, z) - T4_Ex_dy(t, x, y, z)


def T4Bx(t, x, y, z):
    """"""
    return - T4curlE_x1(t, x, y, z) + T4curlE_x1(0, x, y, z)


def T4By(t, x, y, z):
    """"""
    return - T4curlE_y1(t, x, y, z) + T4curlE_y1(0, x, y, z)


def T4Bz(t, x, y, z):
    """"""
    return - T4curlE_z1(t, x, y, z) + T4curlE_z1(0, x, y, z)


class ManufacturedSolution_Hall_MHD3_4_TemporalAccuracy(ManufacturedSolution_Hall_MHD3_0):
    r"""This manufactured solution is for domain [0, 1]^3 x (0, t].

    It is for vanishing b.c. uxn =0, P=0, Bxn = 0 on whole boundary for TEMPORAL convergence test.
    """
    def __init__(self, c=1, Rm=1, Rf=1, eta=1):
        r"""

        Parameters
        ----------
        c :
            Coupling number.
        Rm :
        Rf :

        """
        super().__init__(c=c, Rm=Rm, Rf=Rf, eta=eta)
        self._melt()

        self._u = T3dVector(T4_u, T4_v, T4_w)
        self._P = T3dScalar(T4_P_func)
        self._E = T3dVector(T4_Ex, T4_Ey, T4_Ez)
        self._B = T3dVector(T4Bx, T4By, T4Bz)

        self._freeze()


# ============== CONSERVATION INITIAL CONDITION ========================================================

class ManufacturedSolution_Hall_MHD3_Conservation0(Frozen):
    r"""See Section 4.2 of [Kaibo Hu, Young-Ju Lee, Jinchao Xu, Helicity-conservative finite element discretization
    for incompressible MHD systems, JCP 436 (2021) 110284]

    And this is a modified version of it.

    Basically, we switch x- and y- components of B^0.

    It is used for conservation tests.
    """
    def __init__(self, c=1, Rm=1, Rf=1, eta=1):
        r""""""
        self._c_ = c
        self._Rm_ = Rm
        self._Rf_ = Rf
        self._eta_ = eta
        self._u_init_ = None
        self._w_init_ = None
        self._B_init_ = None
        self._j_init_ = None

        self._E_init_ = None

        self._f_ = None
        self._m_ = None
        self._g_ = None

        self._P_bc_ = None

        self._freeze()

    # noinspection PyUnusedLocal
    @staticmethod
    def _u1_(t, x, y, z):
        """"""
        return - sin(pi * (x - 1/2)) * cos(pi * (y - 1/2)) * z * (z-1)

    # noinspection PyUnusedLocal
    @staticmethod
    def _u2_(t, x, y, z):
        """"""
        return cos(pi * (x - 1/2)) * sin(pi * (y - 1/2)) * z * (z - 1)

    # noinspection PyUnusedLocal
    @staticmethod
    def _u3_(t, x, y, z):
        return zeros_like(x)

    @property
    def u_initial_condition(self):
        if self._u_init_ is None:
            self._u_init_ = T3dVector(self._u1_, self._u2_, self._u3_)
        return self._u_init_

    @property
    def w_initial_condition(self):
        if self._w_init_ is None:
            self._w_init_ = self.u_initial_condition.curl
        return self._w_init_

    # noinspection PyUnusedLocal
    @staticmethod
    def _B0_(t, x, y, z):
        """"""
        return cos(pi * x) * sin(pi * y) * z * (z-1)

    # noinspection PyUnusedLocal
    @staticmethod
    def _B1_(t, x, y, z):
        return - sin(pi * x) * cos(pi * y) * z * (z-1)

    @property
    def B_initial_condition(self):
        if self._B_init_ is None:
            self._B_init_ = T3dVector(self._B0_, self._B1_, 0)
        return self._B_init_

    @property
    def j_initial_condition(self):
        if self._j_init_ is None:
            self._j_init_ = self.B_initial_condition.curl
        return self._j_init_

    @property
    def E_initial_condition(self):
        """E initial condition.

        (1/Rm)j - u x B + eta * j x B = E
        """
        if self._E_init_ is None:
            self._E_init_ = (
                (1 / self._Rm_) * self.j_initial_condition
                - self.u_initial_condition.cross_product(self.B_initial_condition)
                + self._eta_ * (self.j_initial_condition.cross_product(self.B_initial_condition))
            )
        return self._E_init_

    @property
    def P_boundary_condition(self):
        if self._P_bc_ is None:
            self._P_bc_ = T3dScalar(0)
        return self._P_bc_

    @property
    def g(self):
        if self._g_ is None:
            self._g_ = T3dScalar(0)
        return self._g_

    @property
    def f(self):
        if self._f_ is None:
            self._f_ = T3dVector(0, 0, 0)
        return self._f_

    @property
    def m(self):
        if self._m_ is None:
            self._m_ = T3dVector(0, 0, 0)
        return self._m_


# ============== CONSERVATION INITIAL CONDITION ========================================================


class ManufacturedSolution_Hall_MHD3_NullPoints(Frozen):
    r"""See Section 4.4 of [A LINEARLY IMPLICIT SPECTRAL SCHEME FOR THE THREE-DIMENSIONAL HALL-MHD SYSTEM]

    """
    def __init__(self, c=1, Rm=1, Rf=1, eta=1):
        r""""""
        self._c_ = c
        self._Rm_ = Rm
        self._Rf_ = Rf
        self._eta_ = eta
        self._u_init_ = None
        self._w_init_ = None
        self._B_init_ = None
        self._j_init_ = None

        self._E_init_ = None

        self._f_ = None
        self._m_ = None
        self._g_ = None

        self._P_initial_condition_ = None

        self._freeze()

    @property
    def u_initial_condition(self):
        if self._u_init_ is None:
            self._u_init_ = T3dVector(0, 0, 0)
        return self._u_init_

    @property
    def w_initial_condition(self):
        if self._w_init_ is None:
            self._w_init_ = T3dVector(0, 0, 0)
        return self._w_init_

    # noinspection PyUnusedLocal
    @staticmethod
    def _B2_(t, x, y, z):
        return - 2.5 * sin(pi * x)

    @property
    def B_initial_condition(self):
        if self._B_init_ is None:
            self._B_init_ = T3dVector(0, 0, self._B2_)
        return self._B_init_

    @property
    def j_initial_condition(self):
        if self._j_init_ is None:
            self._j_init_ = self.B_initial_condition.curl
        return self._j_init_

    @property
    def E_initial_condition(self):
        """E initial condition.

        (1/Rm)j - u x B + eta * j x B = E
        """
        if self._E_init_ is None:
            self._E_init_ = (
                (1 / self._Rm_) * self.j_initial_condition
                - self.u_initial_condition.cross_product(self.B_initial_condition)
                + self._eta_ * (self.j_initial_condition.cross_product(self.B_initial_condition))
            )
        return self._E_init_

    # noinspection PyUnusedLocal
    @staticmethod
    def ___p___(t, x, y, z):
        r""""""
        return (5 / 4) ** 2 * cos(2 * pi * x)

    @property
    def P_initial_condition(self):
        if self._P_initial_condition_ is None:
            self._P_initial_condition_ = T3dScalar(self.___p___)
        return self._P_initial_condition_

    @property
    def g(self):
        if self._g_ is None:
            self._g_ = T3dScalar(0)
        return self._g_

    @property
    def f(self):
        if self._f_ is None:
            self._f_ = T3dVector(0, 0, 0)
        return self._f_

    @property
    def m(self):
        if self._m_ is None:
            self._m_ = T3dVector(0, 0, 0)
        return self._m_

    @property
    def _f_init_(self):
        """body force."""
        return (
            self.u_initial_condition.time_derivative
            + self.w_initial_condition.cross_product(self.u_initial_condition)
            + (1 / self._Rf_) * self.w_initial_condition.curl
            - self._c_ * (self.j_initial_condition.cross_product(self.B_initial_condition))
            + self.P_initial_condition.gradient
        )


if __name__ == '__main__':
    #
    manu = ManufacturedSolution_Hall_MHD3_NullPoints(1, 1, 1, 1)

    f_init = manu._f_init_
    from random import random
    for _ in range(10):
        x = random() * 2 - 1
        y = random() * 2 - 1
        z = random() * 2 - 1
        value = f_init(0, x, y, z)
        print(value)
