# -*- coding: utf-8 -*-
r"""
"""
import sys
if './' not in sys.path:
    sys.path.append('./')
import numpy as np
from tools.frozen import Frozen
from tools.quadrature import Quadrature
from tools.functions.time_space._2d.wrappers.vector import T2dVector


class ConditionsNormalDipoleCollision2(Frozen):
    """"""

    def __init__(
            self,
            x1=0, y1=0.1, x2=0, y2=-0.1, r0=0.1, we1=320, we2=-320, K_scale=2
    ):
        """"""
        self._x1_ = x1
        self._y1_ = y1
        self._x2_ = x2
        self._y2_ = y2
        self._r0_ = r0
        self._r0_square_ = r0 ** 2
        self._we1_ = we1
        self._we2_ = we2
        assert we1 + we2 == 0
        self._abs_we_ = abs(we1)

        quad_nodes, quad_weights = Quadrature([99, 99], category='Gauss').quad
        quad_nodes = np.meshgrid(*quad_nodes, indexing='ij')

        U0 = self._u_unscaled(*quad_nodes)
        V0 = self._v_unscaled(*quad_nodes)
        K_unscaled = 0.5 * np.einsum(
            'ij, i, j ->', U0**2 + V0**2, quad_weights[0], quad_weights[1], optimize='greedy'
        )

        scale = K_scale / K_unscaled
        self._v_scale_ = np.sqrt(scale)
        self._freeze()

    def _u_unscaled(self, x, y):
        """"""
        x1 = self._x1_
        y1 = self._y1_
        x2 = self._x2_
        y2 = self._y2_

        r1_square = (x - x1) ** 2 + (y - y1) ** 2
        r2_square = (x - x2) ** 2 + (y - y2) ** 2

        term1 = -0.5 * self._abs_we_ * (y - y1) * np.exp(- r1_square / self._r0_square_)
        term2 = 0.5 * self._abs_we_ * (y - y2) * np.exp(- r2_square / self._r0_square_)

        return term1 + term2

    def _v_unscaled(self, x, y):
        x1 = self._x1_
        y1 = self._y1_
        x2 = self._x2_
        y2 = self._y2_

        r1_square = (x - x1) ** 2 + (y - y1) ** 2
        r2_square = (x - x2) ** 2 + (y - y2) ** 2

        term1 = 0.5 * self._abs_we_ * (x - x1) * np.exp(- r1_square / self._r0_square_)
        term2 = -0.5 * self._abs_we_ * (x - x2) * np.exp(- r2_square / self._r0_square_)

        return term1 + term2

    def u(self, t, x, y):
        return self._v_scale_ * self._u_unscaled(x, y) + 0 * t

    def v(self, t, x, y):
        return self._v_scale_ * self._v_unscaled(x, y) + 0 * t

    @property
    def velocity_initial_condition(self):
        """"""
        return T2dVector(self.u, self.v)

    @property
    def vorticity_initial_condition(self):
        return self.velocity_initial_condition.rot

    @property
    def velocity_boundary_condition(self):
        """"""
        return T2dVector(self._0, self._0)

    # noinspection PyUnusedLocal
    @staticmethod
    def _0(t, x, y):
        return 0 * x


if __name__ == '__main__':
    # python tests/samples/conditions_normal_dipole_collision.py
    ic = ConditionsNormalDipoleCollision2()
    ic.vorticity_initial_condition.visualize([-1, 1], 0)
    # ic.velocity_initial_condition.visualize([-1, 1], 0)
