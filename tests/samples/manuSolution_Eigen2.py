# -*- coding: utf-8 -*-
"""
An eigen solution for 2d linear pH system.

Taken from Andrea's Firedrake code.
"""
from numpy import sqrt, sin, cos, pi

from phyem.tools.functions.time_space._2d.wrappers.scalar import T2dScalar
from phyem.tools.functions.time_space._2d.wrappers.vector import T2dVector


class Eigen2:
    """
    """

    def __init__(self, OM=(2*pi, 2*pi), PHI=(0, 0, 0)):
        """"""
        self._OM_ = OM    # om_x, om_y, om_z
        self._PHI_ = PHI   # phi_x, phi_y, phi_z, phi_t

    def p(self, t, x, y):
        """

        Parameters
        ----------
        t
        x
        y

        Returns
        -------

        """
        om_x, om_y = self._OM_
        om_t = sqrt(om_x ** 2 + om_y ** 2)
        phi_x, phi_y, phi_t = self._PHI_

        dft = om_t * (2 * cos(om_t * t + phi_t) - 3 * sin(om_t * t + phi_t))
        g_xy = cos(om_x * x + phi_x) * sin(om_y * y + phi_y)

        return - g_xy * dft

    def u(self, t, x, y):
        om_x, om_y = self._OM_
        om_t = sqrt(om_x ** 2 + om_y ** 2)
        phi_x, phi_y, phi_t = self._PHI_

        ft = 2 * sin(om_t * t + phi_t) + 3 * cos(om_t * t + phi_t)
        dg_xy_x = om_x * sin(om_x * x + phi_x) * sin(om_y * y + phi_y)

        return dg_xy_x * ft

    def v(self, t, x, y):
        om_x, om_y = self._OM_
        om_t = sqrt(om_x ** 2 + om_y ** 2)
        phi_x, phi_y, phi_t = self._PHI_

        ft = 2 * sin(om_t * t + phi_t) + 3 * cos(om_t * t + phi_t)
        dg_xy_y = - om_y * cos(om_x * x + phi_x) * cos(om_y * y + phi_y)

        return dg_xy_y * ft

    @property
    def a(self):
        """da_dt = div b"""
        return self.scalar

    @property
    def b(self):
        """db_dt = grad a"""
        return self.vector

    @property
    def scalar(self):
        """a"""
        return T2dScalar(self.p)

    @property
    def vector(self):
        """b"""
        return T2dVector(self.u, self.v)


if __name__ == '__main__':
    # python tests/samples/manuSolution_Eigen2.py
    eigen = Eigen2()
    # eigen.scalar.visualize([0, 1], 0)
    # eigen.vector.visualize([0, 1], 0)

    a = eigen.scalar
    b = eigen.vector

    # gradient_a = a.gradient
    # db_dt = b.time_derivative
    # gradient_a.visualize([0, 1], 0)
    # db_dt.visualize([0, 1], 0)

    # div_b = b.divergence
    # da_dt = a.time_derivative
    a.visualize([0, 1], 1)
    b.visualize([0, 1], 1)
