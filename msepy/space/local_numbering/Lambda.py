# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.frozen import Frozen


class MsePyLocalNumberingLambda(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._m = space.abstract.m
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
            if self._m == 2 and self._n == 2 and self._k == 1:
                method_name = f"_m{self._m}_n{self._n}_k{self._k}_{self._orientation}"
            else:
                method_name = f"_m{self._m}_n{self._n}_k{self._k}"

            LN = getattr(self, method_name)(p)
            self._cache[key] = LN

        return LN

    @staticmethod
    def _m3_n3_k3(p):
        """"""
        px, py, pz = p
        local_numbering = np.arange(0, px * py * pz).reshape((px, py, pz), order='F')
        return (local_numbering,)  # do not remove (,)

    @staticmethod
    def _m3_n3_k2(p):
        """"""
        px, py, pz = p
        Px = (px+1) * py * pz
        Py = px * (py+1) * pz
        Pz = px * py * (pz+1)
        # faces perp to x-axis
        local_numbering_dy_dz = np.arange(0, Px).reshape((px+1, py, pz), order='F')
        # faces perp to y-axis
        local_numbering_dz_dx = np.arange(Px, Px + Py).reshape((px, py+1, pz), order='F')
        # faces perp to z-axis
        local_numbering_dx_dy = np.arange(Px + Py, Px + Py + Pz).reshape((px, py, pz+1), order='F')
        return local_numbering_dy_dz, local_numbering_dz_dx, local_numbering_dx_dy

    @staticmethod
    def _m3_n3_k1(p):
        """"""
        px, py, pz = p
        Px = px * (py+1) * (pz+1)
        Py = (px+1) * py * (pz+1)
        Pz = (px+1) * (py+1) * pz
        local_numbering_dx = np.arange(0, Px).reshape((px, py+1, pz+1), order='F')
        local_numbering_dy = np.arange(Px, Px + Py).reshape((px+1, py, pz+1), order='F')
        local_numbering_dz = np.arange(Px + Py, Px + Py + Pz).reshape((px+1, py+1, pz), order='F')
        return local_numbering_dx, local_numbering_dy, local_numbering_dz

    @staticmethod
    def _m3_n3_k0(p):
        """"""
        px, py, pz = p
        local_numbering = np.arange(0, (px+1) * (py+1) * (pz+1)).reshape((px+1, py+1, pz+1), order='F')
        return (local_numbering,)  # do not remove (,)

    @staticmethod
    def _m2_n2_k0(p):
        """"""
        px, py = p
        local_numbering = np.arange(0, (px+1) * (py+1)).reshape((px+1, py+1), order='F')
        return (local_numbering,)  # do not remove (,)

    @staticmethod
    def _m2_n2_k1_outer(p):
        """"""
        px, py = p
        Px = (px+1) * py
        Py = px * (py+1)
        # segments perp to x-axis
        local_numbering_dy = np.arange(0, Px).reshape((px+1, py), order='F')
        # segments perp to y-axis
        local_numbering_dx = np.arange(Px, Px + Py).reshape((px, py+1), order='F')
        return local_numbering_dy, local_numbering_dx

    @staticmethod
    def _m2_n2_k1_inner(p):
        """"""
        px, py = p
        Px = px * (py+1)
        Py = (px+1) * py
        local_numbering_dx = np.arange(0, Px).reshape((px, py+1), order='F')
        local_numbering_dy = np.arange(Px, Px + Py).reshape((px+1, py), order='F')
        return local_numbering_dx, local_numbering_dy

    @staticmethod
    def _m2_n2_k2(p):
        """"""
        px, py = p
        local_numbering = np.arange(0, px * py).reshape((px, py), order='F')
        return (local_numbering,)  # do not remove (,)

    @staticmethod
    def _m1_n1_k0(p):
        """"""
        p = p[0]
        local_numbering = np.arange(0, p+1)
        return (local_numbering,)  # do not remove (,)

    @staticmethod
    def _m1_n1_k1(p):
        """"""
        p = p[0]
        local_numbering = np.arange(0, p)
        return (local_numbering,)  # do not remove (,)
