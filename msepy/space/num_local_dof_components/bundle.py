# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen


class MsePyNumLocalDofComponentsBundle(Frozen):
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
        px, py, pz = p[0]
        p0 = px * py * pz
        px, py, pz = p[1]
        p1 = px * py * pz
        px, py, pz = p[2]
        p2 = px * py * pz
        return p0, p1, p2

    @staticmethod
    def _n3_k2(p):
        """"""
        px, py, pz = p[0]
        Px = (px+1) * py * pz
        Py = px * (py+1) * pz
        Pz = px * py * (pz+1)
        p0 = (Px, Py, Pz)

        px, py, pz = p[1]
        Px = (px+1) * py * pz
        Py = px * (py+1) * pz
        Pz = px * py * (pz+1)
        p1 = (Px, Py, Pz)

        px, py, pz = p[2]
        Px = (px+1) * py * pz
        Py = px * (py+1) * pz
        Pz = px * py * (pz+1)
        p2 = (Px, Py, Pz)

        return p0, p1, p2

    @staticmethod
    def _n3_k1(p):
        """"""
        px, py, pz = p[0]
        Px = px * (py+1) * (pz+1)
        Py = (px+1) * py * (pz+1)
        Pz = (px+1) * (py+1) * pz
        p0 = (Px, Py, Pz)

        px, py, pz = p[1]
        Px = px * (py+1) * (pz+1)
        Py = (px+1) * py * (pz+1)
        Pz = (px+1) * (py+1) * pz
        p1 = (Px, Py, Pz)

        px, py, pz = p[2]
        Px = px * (py+1) * (pz+1)
        Py = (px+1) * py * (pz+1)
        Pz = (px+1) * (py+1) * pz
        p2 = (Px, Py, Pz)

        return p0, p1, p2

    @staticmethod
    def _n3_k0(p):
        """"""
        px, py, pz = p[0]
        p0 = (px+1) * (py+1) * (pz+1)
        px, py, pz = p[1]
        p1 = (px+1) * (py+1) * (pz+1)
        px, py, pz = p[2]
        p2 = (px+1) * (py+1) * (pz+1)

        return p0, p1, p2

    @staticmethod
    def _n2_k0(p):
        """"""
        px, py = p[0]
        p0 = (px+1) * (py+1)
        px, py = p[1]
        p1 = (px+1) * (py+1)
        return p0, p1

    @staticmethod
    def _n2_k1_inner(p):
        """"""
        px, py = p[0]
        Px = px * (py+1)
        Py = (px+1) * py
        p0 = Px, Py

        px, py = p[1]
        Px = px * (py+1)
        Py = (px+1) * py
        p1 = Px, Py

        return p0, p1

    @staticmethod
    def _n2_k1_outer(p):
        """"""
        px, py = p[0]
        Px = (px+1) * py
        Py = px * (py+1)
        p0 = Px, Py

        px, py = p[1]
        Px = (px+1) * py
        Py = px * (py+1)
        p1 = Px, Py

        return p0, p1

    @staticmethod
    def _n2_k2(p):
        """"""
        px, py = p[0]
        p0 = px * py
        px, py = p[1]
        p1 = px * py
        return p0, p1

    @staticmethod
    def _n1_k0(p):
        """"""
        p = p[0][0]
        return [p+1, ]

    @staticmethod
    def _n1_k1(p):
        """"""
        p = p[0][0]
        return [p, ]
