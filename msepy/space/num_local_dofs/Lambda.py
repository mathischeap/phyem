# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MsePyNumLocalDofsLambda(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._m = space.abstract.m
        self._freeze()

    def __call__(self, degree):
        """Making the local numbering for degree."""
        p = self._space[degree].p
        return getattr(self, f"_m{self._m}_n{self._n}_k{self._k}")(p)

    @staticmethod
    def _m3_n3_k3(p):
        """"""
        px, py, pz = p
        return px * py * pz

    @staticmethod
    def _m3_n3_k2(p):
        """"""
        px, py, pz = p
        Px = (px+1) * py * pz
        Py = px * (py+1) * pz
        Pz = px * py * (pz+1)
        return Px + Py + Pz

    @staticmethod
    def _m3_n3_k1(p):
        """"""
        px, py, pz = p
        Px = px * (py+1) * (pz+1)
        Py = (px+1) * py * (pz+1)
        Pz = (px+1) * (py+1) * pz
        return Px + Py + Pz

    @staticmethod
    def _m3_n3_k0(p):
        """"""
        px, py, pz = p
        return (px+1) * (py+1) * (pz+1)

    @staticmethod
    def _m2_n2_k0(p):
        """"""
        px, py = p
        return (px+1) * (py+1)

    @staticmethod
    def _m2_n2_k1(p):
        """"""
        px, py = p
        Px = (px+1) * py
        Py = px * (py+1)
        return Px + Py

    @staticmethod
    def _m2_n2_k2(p):
        """"""
        px, py = p
        return px * py

    @staticmethod
    def _m1_n1_k0(p):
        """"""
        p = p[0]
        return p+1

    @staticmethod
    def _m1_n1_k1(p):
        """"""
        p = p[0]
        return p
