# -*- coding: utf-8 -*-
r"""
The shear layer rollup is a 2-dimensional ideal incompressible flow (Euler flow).
And the external body force is zero.

The flow is in a periodic domain :math:`\Omega = [0, 2\pi]^2`, and components of
the initial velocity :math:`\boldsymbol{u}^0 = \begin{bmatrix}u^0 & v^0\end{bmatrix}^{\mathsf{T}}` are

.. math::

    u^0 = \left\lbrace\begin{aligned}
    &\tanh\left(\dfrac{y-\frac{\pi}{2}}{\delta}\right)\quad \text{if }  y\leq\pi\\
    &\tanh\left(\dfrac{\frac{3\pi}{2}-y}{\delta}\right)\quad \text{else}
    \end{aligned}\right.,

and

.. math::
    v^0 = \epsilon\sin(x),

where :math:`\delta=\pi/15` and :math:`\epsilon=0.05`.

"""
from numpy import tanh, sin, pi  # cosh

from phyem.tools.gen_piece_wise import genpiecewise
from phyem.tools.functions.time_space._2d.wrappers.vector import T2dVector
from phyem.tools.frozen import Frozen


class InitialConditionShearLayerRollUp(Frozen):
    """"""

    def __init__(self, delta=pi/15, epsilon=0.05):
        """"""
        self._delta_ = delta
        self._epsilon_ = epsilon
        self._velocity = T2dVector(self.u, self.v)
        self._vorticity = self.velocity.rot
        self._freeze()

    @property
    def epsilon(self):
        return self._epsilon_

    @property
    def delta(self):
        return self._delta_

    # noinspection PyUnusedLocal
    def _u_low_(self, t, x, y):  # y <= pi
        return tanh((y - pi / 2) / self.delta)

    # noinspection PyUnusedLocal
    def _u_up_(self, t, x, y):  # y > pi
        return tanh((3 * pi / 2 - y) / self.delta)

    def u(self, t, x, y):
        return genpiecewise([t, x, y], [y <= pi, y > pi], [self._u_low_, self._u_up_])

    # noinspection PyUnusedLocal
    def v(self, t, x, y):
        """"""
        return self.epsilon * sin(x)

    @property
    def velocity(self):
        """"""
        return self._velocity

    @property
    def vorticity(self):
        return self._vorticity


if __name__ == '__main__':
    # python tests/samples/iniCond_shear_layer_rollup.py
    ic = InitialConditionShearLayerRollUp()
    ic.vorticity.visualize([0, 2*pi], 0)
    ic.velocity.visualize([0, 2*pi], 0)
