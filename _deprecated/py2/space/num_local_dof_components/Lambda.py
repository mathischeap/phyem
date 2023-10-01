# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHyPy2NumLocalDofComponentsLambda(Frozen):
    """Generation in-dependent."""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._k = space.abstract.k
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
            if self._k == 1:
                method_name = f"_k{self._k}_{self._orientation}"
            else:
                method_name = f"_k{self._k}"

            LN = getattr(self, method_name)(p)
            self._cache[key] = LN

        return LN

    @staticmethod
    def _k0(p):
        """"""
        px, py = p
        _q_ = [(px+1) * (py+1), ]

        assert px == py
        _t_ = [px * (py+1) + 1, ]

        return {
            'q': _q_,
            't': _t_,
        }

    @staticmethod
    def _k1_inner(p):
        """"""
        px, py = p
        Px = px * (py+1)
        Py = (px+1) * py
        _q_ = Px, Py

        assert px == py
        Px = px * (py+1)
        Py = px * py
        _t_ = Px, Py

        return {
            'q': _q_,
            't': _t_,
        }

    @staticmethod
    def _k1_outer(p):
        """"""
        px, py = p
        Px = (px+1) * py
        Py = px * (py+1)
        _q_ = Px, Py

        assert px == py
        Px = px * py
        Py = px * (py+1)
        _t_ = Px, Py

        return {
            'q': _q_,
            't': _t_,
        }

    @staticmethod
    def _k2(p):
        """"""
        px, py = p
        _q_ = [px * py, ]

        assert px == py

        return {
            'q': _q_,
            't': _q_,
        }
