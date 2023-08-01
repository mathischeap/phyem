# -*- coding: utf-8 -*-
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
import sys
if './' not in sys.path:
    sys.path.append('./')

from numpy import sin, cos, pi

from tools.frozen import Frozen
from tools.functions.time_space._2d.wrappers.scalar import T2dScalar


def _phi(t, x, y):
    """"""
    return 2 * sin(y) - 2 * cos(x) + 0 * t


def _A(t, x, y):
    """"""
    return cos(2*y) - 2 * cos(x) + 0 * t


class InitialConditionOrszagTangVortex(Frozen):
    """"""
    def __init__(self):
        self._streaming = T2dScalar(_phi)
        self._potential = T2dScalar(_A)
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
    def H(self):
        """magnetic field strength, H = B under non-dimensionalization."""
        return self.B

    @property
    def j(self):
        """electric current density"""
        return self.B.rot

    @property
    def E(self):
        """electric field strength"""
        return - self.u.cross_product(self.B)

    @property
    def omega(self):
        """vorticity"""
        return self.u.rot


if __name__ == '__main__':
    # python tests/samples/iniCond_Orszag_Tang_vortex.py
    ic = InitialConditionOrszagTangVortex()
    ic.j.visualize([0, 2*pi], 0)
