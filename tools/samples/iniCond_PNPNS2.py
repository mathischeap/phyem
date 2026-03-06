# -*- coding: utf-8 -*-
r"""
"""
from numpy import sin, cos, pi

from phyem.tools.frozen import Frozen
from phyem.tools.functions.time_space._2d.wrappers.scalar import T2dScalar
from phyem.tools.functions.time_space._2d.wrappers.vector import T2dVector

# ------------------------------------------------------------------------------------------------
# ------------ #1, energy conservation test, in [0, 1]^2 -----------------------------------------
# ------------------------------------------------------------------------------------------------


# noinspection PyUnusedLocal
def ___u___(t, x, y):
    r""""""
    return pi * (sin(pi * x)) ** 2 * sin(2 * pi * y)


# noinspection PyUnusedLocal
def ___v___(t, x, y):
    r""""""
    return - pi * sin(2 * pi * x) * (sin(pi * y)) ** 2


# noinspection PyUnusedLocal
def ___p___(t, x, y):
    r""""""
    return 1.1 + cos(pi * x) * cos(pi * y)


# noinspection PyUnusedLocal
def ___n___(t, x, y):
    r""""""
    return 1.1 - cos(pi * x) * cos(pi * y)


# noinspection PyUnusedLocal
def ___pmn___(t, x, y):
    r""""""
    return 2 * cos(pi * x) * cos(pi * y)


___cache___ = {
    'a': 0.
}


# noinspection PyUnusedLocal
def ___psi___(t, x, y):
    r""""""
    return ___cache___['a'] * cos(pi * x) * cos(pi * y)


class InitialCondition_PNPNS_2D_Conservation1(Frozen):
    """See Example 2. in Section 5 of [Xiaolan Zhou, Chuanju Xu, 2023, Computer Physics Communications].

    It is an initial condition that has blocking boundary conditions, i.e., vanishing of all normal fluxes
    for the ionic concentrations.
    """

    def __init__(self, epsilon=1, mesh=None):
        """

        Parameters
        ----------
        mesh

        """
        self._epsilon = epsilon
        ___cache___['a'] = 1 / (epsilon * pi ** 2)
        self._mesh = mesh
        self._p = T2dScalar(___p___, mesh=mesh)
        self._n = T2dScalar(___n___, mesh=mesh)
        self._u = T2dVector(___u___, ___v___, mesh=mesh)
        self._psi = None
        self._mu_ = None
        self._nu_ = None
        self._p_grad_mu = None
        self._n_grad_nu = None
        self._tau_ = None
        self._chi_ = None
        self._delta_ = None
        self._theta_ = None
        self._freeze()

    @property
    def p(self):
        r"""concentration of positively charged ions."""
        return self._p

    @property
    def n(self):
        r"""concentration of negatively charged ions."""
        return self._n

    @property
    def psi(self):
        if self._psi is None:
            self._psi = T2dScalar(___psi___, mesh=self._mesh)
        return self._psi

    @property
    def u(self):
        """fluid velocity field."""
        return self._u

    @property
    def omega(self):
        r"""vorticity."""
        return self.u.rot

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
    def p_grad_mu(self):
        r"""p multi grad of mu"""
        if self._p_grad_mu is None:
            self._p_grad_mu = self.p * self.mu.gradient
            self._p_grad_mu._mesh = self._mesh
        return self._p_grad_mu

    @property
    def n_grad_nu(self):
        r"""p multi grad of mu"""
        if self._n_grad_nu is None:
            self._n_grad_nu = self.n * self.nu.gradient
            self._n_grad_nu._mesh = self._mesh
        return self._n_grad_nu

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
