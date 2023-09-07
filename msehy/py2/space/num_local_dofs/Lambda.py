# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHyPy2NumLocalDofsLambda(Frozen):
    """Generation in-dependent."""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._k = space.abstract.k
        self._freeze()

    def __call__(self, degree):
        """Making the local numbering for degree."""
        p = self._space[degree].p
        return getattr(self, f"_k{self._k}")(p)

    @staticmethod
    def _k0(p):
        """"""
        px, py = p
        _q_ = (px+1) * (py+1)
        assert px == py
        _t_ = px * (py+1) + 1
        return {
            'q': _q_,
            't': _t_,
        }

    @staticmethod
    def _k1(p):
        """"""
        px, py = p
        Px = (px+1) * py
        Py = px * (py+1)
        _q_ = Px + Py
        assert px == py
        _t_ = px * py + px * (py+1)

        return {
            'q': _q_,
            't': _t_,
        }

    @staticmethod
    def _k2(p):
        """"""
        px, py = p
        _q_ = px * py
        assert px == py
        return {
            'q': _q_,
            't': _q_,
        }
