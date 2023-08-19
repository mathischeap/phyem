# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MsePyNumLocalDofComponentsLambda(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
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
        px, py, pz = p
        return [px * py * pz, ]

    @staticmethod
    def _n3_k2(p):
        """"""
        px, py, pz = p
        Px = (px+1) * py * pz
        Py = px * (py+1) * pz
        Pz = px * py * (pz+1)
        return Px, Py, Pz

    @staticmethod
    def _n3_k1(p):
        """"""
        px, py, pz = p
        Px = px * (py+1) * (pz+1)
        Py = (px+1) * py * (pz+1)
        Pz = (px+1) * (py+1) * pz
        return Px, Py, Pz

    @staticmethod
    def _n3_k0(p):
        """"""
        px, py, pz = p
        return [(px+1) * (py+1) * (pz+1), ]

    @staticmethod
    def _n2_k0(p):
        """"""
        px, py = p
        return [(px+1) * (py+1), ]

    @staticmethod
    def _n2_k1_inner(p):
        """"""
        px, py = p
        Px = px * (py+1)
        Py = (px+1) * py
        return Px, Py

    @staticmethod
    def _n2_k1_outer(p):
        """"""
        px, py = p
        Px = (px+1) * py
        Py = px * (py+1)
        return Px, Py

    @staticmethod
    def _n2_k2(p):
        """"""
        px, py = p
        return [px * py, ]

    @staticmethod
    def _n1_k0(p):
        """"""
        p = p[0]
        return [p+1, ]

    @staticmethod
    def _n1_k1(p):
        """"""
        p = p[0]
        return [p, ]
