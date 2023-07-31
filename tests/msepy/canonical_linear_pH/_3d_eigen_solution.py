# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 4:03 PM on 5/22/2023
"""

from numpy import sqrt, sin, cos, pi


class Eigen1:
    """The first eigen-solution.
    """

    def __init__(self, OM=(2*pi, 2*pi, 2*pi), PHI=(0, 0, 0, 0)):
        """"""
        self._OM_ = OM    # om_x, om_y, om_z
        self._PHI_ = PHI   # phi_x, phi_y, phi_z, phi_t

    def p(self, t, x, y, z):
        """Translated from Andrea's Firedrake codes.

        Parameters
        ----------
        t
        x
        y
        z

        Returns
        -------

        """
        om_x, om_y, om_z = self._OM_
        om_t = sqrt(om_x ** 2 + om_y ** 2 + om_z ** 2)
        phi_x, phi_y, phi_z, phi_t = self._PHI_

        dft = om_t * (2 * cos(om_t * t + phi_t) - 3 * sin(om_t * t + phi_t))
        g_xyz = cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z)

        return - g_xyz * dft

    def u(self, t, x, y, z):
        om_x, om_y, om_z = self._OM_
        om_t = sqrt(om_x ** 2 + om_y ** 2 + om_z ** 2)
        phi_x, phi_y, phi_z, phi_t = self._PHI_

        ft = 2 * sin(om_t * t + phi_t) + 3 * cos(om_t * t + phi_t)
        dg_xyz_x = - om_x * sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z)

        return dg_xyz_x * ft

    def v(self, t, x, y, z):
        om_x, om_y, om_z = self._OM_
        om_t = sqrt(om_x ** 2 + om_y ** 2 + om_z ** 2)
        phi_x, phi_y, phi_z, phi_t = self._PHI_

        ft = 2 * sin(om_t * t + phi_t) + 3 * cos(om_t * t + phi_t)
        dg_xyz_y = om_y * cos(om_x * x + phi_x) * cos(om_y * y + phi_y) * sin(om_z * z + phi_z)

        return dg_xyz_y * ft

    def w(self, t, x, y, z):
        om_x, om_y, om_z = self._OM_
        om_t = sqrt(om_x ** 2 + om_y ** 2 + om_z ** 2)
        phi_x, phi_y, phi_z, phi_t = self._PHI_

        ft = 2 * sin(om_t * t + phi_t) + 3 * cos(om_t * t + phi_t)
        dg_xyz_z = om_z * cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * cos(om_z * z + phi_z)

        return dg_xyz_z * ft
