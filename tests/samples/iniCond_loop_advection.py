# -*- coding: utf-8 -*-
r"""
"""
import sys
if './' not in sys.path:
    sys.path.append('./')

import numpy as np

from tools.frozen import Frozen
from tools.functions.time_space._2d.wrappers.scalar import T2dScalar
from tools.functions.time_space._2d.wrappers.vector import T2dVector

from tools.gen_piece_wise import genpiecewise


class InitialCondition_LoopAdvection(Frozen):
    """"""
    def __init__(self, Lx=2, Ly=1, A0=1e-3, R=0.3):
        self._Lx = Lx
        self._Ly = Ly
        self._A0 = A0
        self._R = R
        self._theta = np.arctan(Ly / Lx)
        self._V0 = np.sqrt(Lx**2 + Ly**2)
        self._potential = T2dScalar(self._A)
        self._freeze()

    def _A_in_radius(self, x, y):
        """"""
        return self._A0 * (self._R - np.sqrt(x ** 2 + y ** 2))

    @staticmethod
    def _A_out_radius(x, y):
        """"""
        return np.zeros_like(x) + np.zeros_like(y)

    def _A(self, t, x, y):
        """"""
        r = np.sqrt(x ** 2 + y ** 2)
        A = genpiecewise(
            [x, y],
            [r <= self._R, r > self._R],
            [self._A_in_radius, self._A_out_radius]
        )
        return A + 0 * t

    def _Vx(self, t, x, y):
        """"""
        return self._V0 * np.cos(self._theta) + np.zeros_like(x) + np.zeros_like(y) + 0 * t

    def _Vy(self, t, x, y):
        """"""
        return self._V0 * np.sin(self._theta) + np.zeros_like(x) + np.zeros_like(y) + 0 * t

    @property
    def u(self):
        """fluid velocity field"""
        return T2dVector(self._Vx, self._Vy)

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
    def omega(self):
        """vorticity"""
        return self.u.rot


if __name__ == '__main__':
    # python tests/samples/iniCond_loop_advection.py
    ic = InitialCondition_LoopAdvection()
    ic.B.norm.visualize(
        [[-1, 1], [-0.5, 0.5]], 0,
        sampling_factor=10,
        plot_type='contour'
    )
