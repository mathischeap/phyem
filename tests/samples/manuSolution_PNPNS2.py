# -*- coding: utf-8 -*-
r"""
"""
import sys

if './' not in sys.path:
    sys.path.append('./')

from numpy import sin, cos, exp
from tools.frozen import Frozen

from tools.functions.time_space._2d.wrappers.scalar import T2dScalar
from tools.functions.time_space._2d.wrappers.vector import T2dVector


# ------------------------------------------------------------------------------------------------
# ------------ #1, periodic, in [0, 2pi]^2 -------------------------------------------------------
# ------------------------------------------------------------------------------------------------


class Manufactured_Solution_PNPNS_2D_PeriodicDomain1(Frozen):
    """The Domain must be [0, 2pi]^2 and is periodic."""

    def __init__(self):
        """

        Parameters
        ----------

        """
        self._epsilon = 1
        self._velocity = T2dVector(
            self._u, self._v,
            Jacobian_matrix=(
                [self._ux, self._uy],
                [self._vx, self._vy]
            ),
            time_derivative=[self._ut, self._vt]
        )
        self._static_pressure = T2dScalar(  # static pressure
            self._sp,
            derivative=[self._sp_t, self._sp_x, self._sp_y],
        )
        self._pci = T2dScalar(  # concentration of positively charged ions
            self._p,
            derivative=[self._pt, self._px, self._py],
        )
        self._esp = T2dScalar(  # electrostatic potential
            self._psi,
            derivative=[self._psi_t, self._psi_x, self._psi_y],
            second_derivative=[None, self._psi_xx, self._psi_yy, None]
        )

        self._n = None                    # concentration of positively charged ions
        self._mu_ = None                  # mu = ln p + psi
        self._nu_ = None                  # nu = ln n - psi
        self._delta_ = None               # delta = ln p
        self._theta_ = None               # theta = ln n
        self._tau_ = None                 # tau = d(mu) = mu.gradient
        self._chi_ = None                 # chi = d(nu) = nu.gradient
        self._phi_ = None                 # modified pressure = total pressure - p - n

        self._freeze()

    @staticmethod
    def _u(t, x, y):
        r""""""
        return sin(x) * cos(y) * exp(t)

    @staticmethod
    def _ut(t, x, y):
        r""""""
        return sin(x) * cos(y) * exp(t)

    @staticmethod
    def _ux(t, x, y):
        r""""""
        return cos(x) * cos(y) * exp(t)

    @staticmethod
    def _uy(t, x, y):
        r""""""
        return sin(x) * cos(y) * exp(t)

    @staticmethod
    def _v(t, x, y):
        r""""""
        return - cos(x) * sin(y) * exp(t)

    @staticmethod
    def _vt(t, x, y):
        r""""""
        return - cos(x) * sin(y) * exp(t)

    @staticmethod
    def _vx(t, x, y):
        r""""""
        return sin(x) * sin(y) * exp(t)

    @staticmethod
    def _vy(t, x, y):
        r""""""
        return - cos(x) * cos(y) * exp(t)

    @staticmethod
    def _sp(t, x, y):
        r"""Static pressure."""
        return sin(x) * sin(y) * exp(t)

    @staticmethod
    def _sp_t(t, x, y):
        r"""d/dt of Static pressure."""
        return sin(x) * sin(y) * exp(t)

    @staticmethod
    def _sp_x(t, x, y):
        r"""d/dx of Static pressure."""
        return cos(x) * sin(y) * exp(t)

    @staticmethod
    def _sp_y(t, x, y):
        r"""d/dy of Static pressure."""
        return sin(x) * cos(y) * exp(t)

    @staticmethod
    def _p(t, x, y):
        r"""concentration of positively charged ions."""
        return cos(x) * sin(y) * exp(t) + 100

    @staticmethod
    def _pt(t, x, y):
        r"""d/dt of concentration of positively charged ions."""
        return cos(x) * sin(y) * exp(t)

    @staticmethod
    def _px(t, x, y):
        r"""d/dx of concentration of positively charged ions."""
        return - sin(x) * sin(y) * exp(t)

    @staticmethod
    def _py(t, x, y):
        r"""d/dy of concentration of positively charged ions."""
        return cos(x) * cos(y) * exp(t)

    @staticmethod
    def _psi(t, x, y):
        r"""electrostatic potential"""
        return sin(x) * sin(y) * exp(t)

    @staticmethod
    def _psi_t(t, x, y):
        r"""d/dt of electrostatic potential"""
        return sin(x) * sin(y) * exp(t)

    @staticmethod
    def _psi_x(t, x, y):
        r"""d/dx of electrostatic potential"""
        return cos(x) * sin(y) * exp(t)

    @staticmethod
    def _psi_xx(t, x, y):
        r"""d/dx of electrostatic potential"""
        return - sin(x) * sin(y) * exp(t)

    @staticmethod
    def _psi_y(t, x, y):
        r"""d/dy of electrostatic potential"""
        return sin(x) * cos(y) * exp(t)

    @staticmethod
    def _psi_yy(t, x, y):
        r"""d/dy of electrostatic potential"""
        return - sin(x) * sin(y) * exp(t)

    @property
    def p(self):
        r"""concentration of positively charged ions."""
        return self._pci

    @property
    def n(self):
        r"""concentration of negatively charged ions."""
        if self._n is None:
            self._n = self.p + self._epsilon * self.psi.Laplacian
        return self._n

    @property
    def psi(self):
        r"""electrostatic potential"""
        return self._esp

    @property
    def mu(self):
        r"""mu = ln p + psi"""
        if self._mu_ is None:
            self._mu_ = self.p.log() + self.psi
        return self._mu_

    @property
    def nu(self):
        r"""nu = ln n - psi"""
        if self._nu_ is None:
            self._nu_ = self.n.log() - self.psi
        return self._nu_

    @property
    def delta(self):
        r"""mu = ln p + psi"""
        if self._delta_ is None:
            self._delta_ = self.p.log()
        return self._delta_

    @property
    def theta(self):
        r"""nu = ln n - psi"""
        if self._theta_ is None:
            self._theta_ = self.n.log()
        return self._theta_

    @property
    def tau(self):
        r"""mu = ln p + psi"""
        if self._tau_ is None:
            self._tau_ = self.mu.gradient
        return self._tau_

    @property
    def chi(self):
        r"""nu = ln n - psi"""
        if self._chi_ is None:
            self._chi_ = self.nu.gradient
        return self._chi_

    @property
    def u(self):
        """fluid velocity field"""
        return self._velocity

    @property
    def omega(self):
        return self.u.rot

    @property
    def phi(self):
        r"""modified pressure = total pressure - p - n"""
        if self._phi_ is None:
            self._phi_ = self.total_pressure - self.p - self.n
        return self._phi_

    @property
    def static_pressure(self):
        return self._static_pressure

    @property
    def total_pressure(self):
        return self.static_pressure + 0.5 * (self.u.dot(self.u))

    @property
    def div_u(self):
        r"""divergence of velocity."""
        return self.u.divergence

    @property
    def source_f(self):
        """source term of momentum equation; body force."""
        return (self.u.time_derivative + self.u.convection_by(self.u) - self.u.Laplacian
                + self.static_pressure.gradient + (self.p - self.n) * self.psi.gradient)

    @property
    def source_p(self):
        r"""source term of p-evolution equation."""
        return (self.p.time_derivative + self.u.dot(self.p.gradient) -
                (self.p.gradient + self.p * self.psi.gradient).divergence)

    @property
    def source_n(self):
        r"""source term of p-evolution equation."""
        return (self.n.time_derivative + self.u.dot(self.n.gradient) -
                (self.n.gradient - self.n * self.psi.gradient).divergence)

# ====================================================================================================
