# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
r"""
The Orszag-Tang Vortex is a 2d non-dimensional MHD manufactured test.

The domain is :math:`(x, y) = [0, 2\pi]^2`. The initial conditions of the velocity and the magnetic field
are

.. math::
    \boldsymbol{u} = \nabla\times \varphi \quad \text{and} \quad \boldsymbol{B} = \nabla\times A,

where :math:`\varphi = 2\sin(y) - 2\cos(x)` and :math:`A=\cos(2y) - 2 \cos(x)`.
See Section 5.3 of
`[Michael Kraus and Omar Maj, Variational Integrators for Ideal Magnetohydrodynamics, 2021, arXiv]
<https://arxiv.org/abs/1707.03227>`_

.. testsetup:: *

    import __init__ as ph

.. testcleanup::

    pass

These initial conditions are implemented. To initialize an ideal Orszag-Tang Vortex, for example,  do

>>> from math import inf
>>> ph.samples.InitialConditionOrszagTangVortex()  # doctest: +ELLIPSIS
<tests.samples.iniCond_Orszag_Tang_vortex.InitialConditionOrszagTangVortex object at ...

"""
from numpy import sin, cos, pi, inf

from phyem.tools.frozen import Frozen
from phyem.tools.functions.time_space._2d.wrappers.scalar import T2dScalar
from phyem.tools.functions.time_space._2d.wrappers.vector import T2dVector
from phyem.tools.functions.time_space._3d.wrappers.vector import T3dVector


def _phi(t, x, y):
    """"""
    return 2 * sin(y) - 2 * cos(x) + 0 * t


def _A(t, x, y):
    """"""
    return cos(2*y) - 2 * cos(x) + 0 * t


class InitialConditionOrszagTangVortex(Frozen):
    """In a periodic domain [0, 2pi]^2."""
    def __init__(self, Rm=inf):
        self._streaming = T2dScalar(_phi)
        self._potential = T2dScalar(_A)
        self._Rm = Rm
        self._freeze()

    @property
    def u(self):
        """fluid velocity field"""
        return self._streaming.curl
    
    @property
    def magnetic_potential(self):
        return self._potential
    
    @property
    def B(self):
        """magnetic flux density"""
        return self._potential.curl

    @property
    def H(self):
        """magnetic field strength, H = B under nondimensionalization."""
        return self.B

    @property
    def j(self):
        """electric current density"""
        return self.B.rot

    @property
    def E(self):
        """electric field strength"""
        return (1 / self._Rm) * self.j - self.u.cross_product(self.B)

    @property
    def omega(self):
        """vorticity"""
        return self.u.rot

    @property
    def f(self):
        return T2dVector(0, 0)


# noinspection PyUnusedLocal
def _phi_3d(t, x, y, z):
    """"""
    return 2 * sin(y) - 2 * cos(x)


# noinspection PyUnusedLocal
def _phi_3d_dx(t, x, y, z):
    """"""
    return 2 * sin(x)


# noinspection PyUnusedLocal
def _phi_3d_dy(t, x, y, z):
    """"""
    return 2 * cos(y)


# noinspection PyUnusedLocal
def _A_3d(t, x, y, z):
    """"""
    return cos(2*y) - 2 * cos(x)


# noinspection PyUnusedLocal
def _A_3d_dx(t, x, y, z):
    """"""
    return 2 * sin(x)


# noinspection PyUnusedLocal
def _A_3d_dy(t, x, y, z):
    """"""
    return - 2 * sin(2*y)


class InitialConditionOrszagTangVortex_3D(Frozen):
    """In a periodic domain [0, 2pi]^2 X [a, b]."""

    def __init__(self, Rm=inf, eta=1):
        self._streaming = T3dVector(
            0, 0, _phi_3d,
            steady=True,
            Jacobian_matrix=(
                [0, 0, 0],
                [0, 0, 0],
                [_phi_3d_dx, _phi_3d_dy, 0],
            ),
        )
        self._potential = T3dVector(
            0, 0, _A_3d,
            steady=True,
            Jacobian_matrix=(
                [0, 0, 0],
                [0, 0, 0],
                [_A_3d_dx, _A_3d_dy, 0],
            ),
        )
        self._Rm = Rm
        self._eta = eta  # parameters of the HALL-EFFECT term.
        self._E_init_ = None
        self._freeze()

    @property
    def u(self):
        """fluid velocity field"""
        return self._streaming.curl

    @property
    def magnetic_potential(self):
        return self._potential

    @property
    def B(self):
        """magnetic flux density"""
        return self._potential.curl

    @property
    def H(self):
        """magnetic field strength, H = B under nondimensionalization."""
        return self.B

    @property
    def j(self):
        """electric current density"""
        return self.B.curl

    @property
    def E(self):
        """electric field strength

        (1/Rm)j - u x B + eta * j x B = E
        """
        if self._E_init_ is None:
            self._E_init_ = (
                (1 / self._Rm) * self.j
                - self.u.cross_product(self.B)
                + self._eta * (self.j.cross_product(self.B))
            )
        return self._E_init_

    @property
    def omega(self):
        """vorticity"""
        return self.u.curl

    @property
    def f(self):
        return T3dVector(0, 0, 0)

    @property
    def zero(self):
        return T3dVector(0, 0, 0)


if __name__ == '__main__':
    # python tests/samples/iniCond_Orszag_Tang_vortex.py
    ic = InitialConditionOrszagTangVortex(Rm=100)
    ic.j.visualize([0, 2*pi], 0)
