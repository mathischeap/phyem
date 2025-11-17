# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen


class MsePyNumLocalDofsBundle(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._freeze()

    def __call__(self, degree):
        """Making the local numbering for degree."""
        p = self._space[degree].p
        return getattr(self, f"_n{self._n}_k{self._k}")(p)

    @staticmethod
    def _n3_k3(p):
        """"""
        px, py, pz = p[0]
        p0 = px * py * pz
        px, py, pz = p[1]
        p1 = px * py * pz
        px, py, pz = p[2]
        p2 = px * py * pz
        return p0 + p1 + p2

    @staticmethod
    def _n3_k2(p):
        """"""
        px, py, pz = p[0]
        Px = (px+1) * py * pz
        Py = px * (py+1) * pz
        Pz = px * py * (pz+1)
        p0 = Px + Py + Pz

        px, py, pz = p[1]
        Px = (px+1) * py * pz
        Py = px * (py+1) * pz
        Pz = px * py * (pz+1)
        p1 = Px + Py + Pz

        px, py, pz = p[2]
        Px = (px+1) * py * pz
        Py = px * (py+1) * pz
        Pz = px * py * (pz+1)
        p2 = Px + Py + Pz
        return p0 + p1 + p2

    @staticmethod
    def _n3_k1(p):
        """"""
        px, py, pz = p[0]
        Px = px * (py+1) * (pz+1)
        Py = (px+1) * py * (pz+1)
        Pz = (px+1) * (py+1) * pz
        p0 = Px + Py + Pz

        px, py, pz = p[1]
        Px = px * (py+1) * (pz+1)
        Py = (px+1) * py * (pz+1)
        Pz = (px+1) * (py+1) * pz
        p1 = Px + Py + Pz

        px, py, pz = p[2]
        Px = px * (py+1) * (pz+1)
        Py = (px+1) * py * (pz+1)
        Pz = (px+1) * (py+1) * pz
        p2 = Px + Py + Pz

        return p0 + p1 + p2

    @staticmethod
    def _n3_k0(p):
        """"""
        px, py, pz = p[0]
        p0 = (px+1) * (py+1) * (pz+1)
        px, py, pz = p[1]
        p1 = (px+1) * (py+1) * (pz+1)
        px, py, pz = p[2]
        p2 = (px+1) * (py+1) * (pz+1)
        return p0 + p1 + p2

    @staticmethod
    def _n2_k0(p):
        """"""
        px, py = p[0]
        p0 = (px+1) * (py+1)
        px, py = p[1]
        p1 = (px+1) * (py+1)
        return p0 + p1

    @staticmethod
    def _n2_k1(p):
        """"""
        px, py = p[0]
        Px = (px+1) * py
        Py = px * (py+1)
        p0 = Px + Py

        px, py = p[1]
        Px = (px+1) * py
        Py = px * (py+1)
        p1 = Px + Py

        return p0 + p1

    @staticmethod
    def _n2_k2(p):
        """"""
        px, py = p[0]
        p0 = px * py
        px, py = p[1]
        p1 = px * py
        return p0 + p1

    @staticmethod
    def _n1_k0(p):
        """"""
        return p[0][0]+1

    @staticmethod
    def _n1_k1(p):
        """"""
        return p[0][0]
