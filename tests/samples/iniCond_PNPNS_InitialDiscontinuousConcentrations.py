# -*- coding: utf-8 -*-
r"""
See [Mehdi Dehghan · Zeinab Gharibi · Ricardo Ruiz-Baier, Journal of Scientific Computing (2023) 94:72, Section 6.2]
"""
import numpy as np

from phyem.tools.frozen import Frozen
from phyem.tools.functions.time_space._2d.wrappers.scalar import T2dScalar
from phyem.tools.functions.time_space._2d.wrappers.vector import T2dVector

from phyem.tools.gen_piece_wise import genpiecewise


def ___one___(x, y):
    r""""""
    return np.ones_like(x)


def ___zero___(x, y):
    r""""""
    return 1e-6 * np.ones_like(x)


def ___p0___(t, x, y):
    r""""""
    return genpiecewise(
        [x, y],
        [x < 0.75, np.logical_and(x >= 0.75, y < 11/20), np.logical_and(x >= 0.75, y >= 11/20)],
        [___zero___, ___zero___, ___one___]
    )


def ___n0___(t, x, y):
    r""""""
    return genpiecewise(
        [x, y],
        [x < 0.75, np.logical_and(x >= 0.75, y > 9/20), np.logical_and(x >= 0.75, y <= 9/20)],
        [___zero___, ___zero___, ___one___]
    )


class Manufactured_Solution_PNPNS_2D_InitialDiscontinuousConcentrations(Frozen):
    """The Domain must be [0, 2pi]^2 and is periodic."""

    def __init__(self, epsilon=1, mesh=None):
        """

        Parameters
        ----------

        """
        self._epsilon = epsilon
        self._p = T2dScalar(___p0___, mesh=mesh)
        self._n = T2dScalar(___n0___, mesh=mesh)
        self._u = T2dVector(0, 0)
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
    def u(self):
        """fluid velocity field."""
        return self._u

    @property
    def omega(self):
        r"""vorticity."""
        return self.u.rot
