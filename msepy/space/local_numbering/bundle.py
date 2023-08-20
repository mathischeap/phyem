# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import numpy as np


class MsePyLocalNumberingBundle(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._orientation = space.abstract.orientation
        self._cache = dict()
        self._freeze()

    def __call__(self, degree):
        """Making the local numbering for degree."""
        p = self._space[degree].p

        key = f"{p}"

        if key in self._cache:
            LN = self._cache[key]
        else:
            if self._n == 2 and self._k == 1:
                method_name = f"_n{self._n}_k{self._k}_{self._orientation}"
            else:
                method_name = f"_n{self._n}_k{self._k}"

            LN = getattr(self, method_name)(p)
            self._cache[key] = LN

        return LN

    @staticmethod
    def _n3_k3(p):
        """"""
        _ = 0
        px, py, pz = p[0]
        x_local_numbering = np.arange(_, _ + px * py * pz).reshape((px, py, pz), order='F')

        _ += px * py * pz
        px, py, pz = p[1]
        y_local_numbering = np.arange(_, _ + px * py * pz).reshape((px, py, pz), order='F')

        _ += px * py * pz
        px, py, pz = p[2]
        z_local_numbering = np.arange(_, _ + px * py * pz).reshape((px, py, pz), order='F')

        return x_local_numbering, y_local_numbering, z_local_numbering

    @staticmethod
    def _n3_k2(p):
        """"""
        _ = 0
        px, py, pz = p[0]
        Px = (px+1) * py * pz
        Py = px * (py+1) * pz
        Pz = px * py * (pz+1)
        # faces perp to x-axis
        local_numbering_dy_dz = np.arange(_, _ + Px).reshape((px+1, py, pz), order='F')
        # faces perp to y-axis
        local_numbering_dz_dx = np.arange(_ + Px, _ + Px + Py).reshape((px, py+1, pz), order='F')
        # faces perp to z-axis
        local_numbering_dx_dy = np.arange(_ + Px + Py, _ + Px + Py + Pz).reshape((px, py, pz+1), order='F')
        x_numbering = (local_numbering_dy_dz, local_numbering_dz_dx, local_numbering_dx_dy)

        _ += Px + Py + Pz
        px, py, pz = p[1]
        Px = (px+1) * py * pz
        Py = px * (py+1) * pz
        Pz = px * py * (pz+1)
        # faces perp to x-axis
        local_numbering_dy_dz = np.arange(_, _ + Px).reshape((px+1, py, pz), order='F')
        # faces perp to y-axis
        local_numbering_dz_dx = np.arange(_ + Px, _ + Px + Py).reshape((px, py+1, pz), order='F')
        # faces perp to z-axis
        local_numbering_dx_dy = np.arange(_ + Px + Py, _ + Px + Py + Pz).reshape((px, py, pz+1), order='F')
        y_numbering = (local_numbering_dy_dz, local_numbering_dz_dx, local_numbering_dx_dy)

        _ += Px + Py + Pz
        px, py, pz = p[2]
        Px = (px+1) * py * pz
        Py = px * (py+1) * pz
        Pz = px * py * (pz+1)
        # faces perp to x-axis
        local_numbering_dy_dz = np.arange(_, _ + Px).reshape((px+1, py, pz), order='F')
        # faces perp to y-axis
        local_numbering_dz_dx = np.arange(_ + Px, _ + Px + Py).reshape((px, py+1, pz), order='F')
        # faces perp to z-axis
        local_numbering_dx_dy = np.arange(_ + Px + Py, _ + Px + Py + Pz).reshape((px, py, pz+1), order='F')
        z_numbering = (local_numbering_dy_dz, local_numbering_dz_dx, local_numbering_dx_dy)

        return x_numbering, y_numbering, z_numbering

    @staticmethod
    def _n3_k1(p):
        """"""
        _ = 0
        px, py, pz = p[0]
        Px = px * (py+1) * (pz+1)
        Py = (px+1) * py * (pz+1)
        Pz = (px+1) * (py+1) * pz
        local_numbering_dx = np.arange(_, _ + Px).reshape((px, py+1, pz+1), order='F')
        local_numbering_dy = np.arange(_ + Px, _ + Px + Py).reshape((px+1, py, pz+1), order='F')
        local_numbering_dz = np.arange(_ + Px + Py, _ + Px + Py + Pz).reshape((px+1, py+1, pz), order='F')
        x_numbering = (local_numbering_dx, local_numbering_dy, local_numbering_dz)

        _ += Px + Py + Pz
        px, py, pz = p[1]
        Px = px * (py+1) * (pz+1)
        Py = (px+1) * py * (pz+1)
        Pz = (px+1) * (py+1) * pz
        local_numbering_dx = np.arange(_, _ + Px).reshape((px, py+1, pz+1), order='F')
        local_numbering_dy = np.arange(_ + Px, _ + Px + Py).reshape((px+1, py, pz+1), order='F')
        local_numbering_dz = np.arange(_ + Px + Py, _ + Px + Py + Pz).reshape((px+1, py+1, pz), order='F')
        y_numbering = (local_numbering_dx, local_numbering_dy, local_numbering_dz)

        _ += Px + Py + Pz
        px, py, pz = p[2]
        Px = px * (py+1) * (pz+1)
        Py = (px+1) * py * (pz+1)
        Pz = (px+1) * (py+1) * pz
        local_numbering_dx = np.arange(_, _ + Px).reshape((px, py+1, pz+1), order='F')
        local_numbering_dy = np.arange(_ + Px, _ + Px + Py).reshape((px+1, py, pz+1), order='F')
        local_numbering_dz = np.arange(_ + Px + Py, _ + Px + Py + Pz).reshape((px+1, py+1, pz), order='F')
        z_numbering = (local_numbering_dx, local_numbering_dy, local_numbering_dz)

        return x_numbering, y_numbering, z_numbering

    @staticmethod
    def _n3_k0(p):
        """"""
        _ = 0
        px, py, pz = p[0]
        x_local_numbering = np.arange(_, _ + (px+1) * (py+1) * (pz+1)).reshape(
            (px+1, py+1, pz+1), order='F')

        _ += (px+1) * (py+1) * (pz+1)
        px, py, pz = p[1]
        y_local_numbering = np.arange(_, _ + (px+1) * (py+1) * (pz+1)).reshape(
            (px+1, py+1, pz+1), order='F')

        _ += (px+1) * (py+1) * (pz+1)
        px, py, pz = p[2]
        z_local_numbering = np.arange(_, _ + (px+1) * (py+1) * (pz+1)).reshape(
            (px+1, py+1, pz+1), order='F')

        return x_local_numbering, y_local_numbering, z_local_numbering

    @staticmethod
    def _n2_k0(p):
        """"""
        px, py = p[0]
        local_numbering_dx = np.arange(0, (px+1) * (py+1)).reshape((px+1, py+1), order='F')
        _ = (px+1) * (py+1)

        px, py = p[1]
        local_numbering_dy = np.arange(_, _ + (px+1) * (py+1)).reshape((px+1, py+1), order='F')
        return local_numbering_dx, local_numbering_dy

    @staticmethod
    def _n2_k1_inner(p):
        """"""
        px, py = p[0]
        Px = px * (py+1)
        Py = (px+1) * py
        x_local_numbering_dx = np.arange(0, Px).reshape((px, py+1), order='F')
        x_local_numbering_dy = np.arange(Px, Px + Py).reshape((px+1, py), order='F')

        _ = Px + Py

        px, py = p[1]
        Px = px * (py+1)
        Py = (px+1) * py
        y_local_numbering_dx = np.arange(_, _ + Px).reshape((px, py+1), order='F')
        y_local_numbering_dy = np.arange(_ + Px, _ + Px + Py).reshape((px+1, py), order='F')

        return (
            (x_local_numbering_dx, x_local_numbering_dy),
            (y_local_numbering_dx, y_local_numbering_dy),
        )

    @staticmethod
    def _n2_k1_outer(p):
        """"""
        px, py = p[0]
        Px = (px+1) * py
        Py = px * (py+1)
        # segments perp to x-axis
        x_local_numbering_dy = np.arange(0, Px).reshape((px+1, py), order='F')
        # segments perp to y-axis
        x_local_numbering_dx = np.arange(Px, Px + Py).reshape((px, py+1), order='F')

        _ = Px + Py

        px, py = p[1]
        Px = (px+1) * py
        Py = px * (py+1)
        # segments perp to x-axis
        y_local_numbering_dy = np.arange(_,  _ + Px).reshape((px+1, py), order='F')
        # segments perp to y-axis
        y_local_numbering_dx = np.arange(_ + Px, _ + Px + Py).reshape((px, py+1), order='F')

        return (
            (x_local_numbering_dy, x_local_numbering_dx),
            (y_local_numbering_dy, y_local_numbering_dx),
        )

    @staticmethod
    def _n2_k2(p):
        """"""
        px, py = p[0]
        local_numbering_dx = np.arange(0, px * py).reshape((px, py), order='F')
        _ = px * py

        px, py = p[1]
        local_numbering_dy = np.arange(_, _ + px * py).reshape((px, py), order='F')
        return local_numbering_dx, local_numbering_dy

    @staticmethod
    def _n1_k0(p):
        """"""
        p = p[0][0]
        local_numbering = np.arange(0, p+1)
        return (local_numbering,)  # do not remove (,)

    @staticmethod
    def _n1_k1(p):
        """"""
        p = p[0][0]
        local_numbering = np.arange(0, p)
        return (local_numbering,)  # do not remove (,)
