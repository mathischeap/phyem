# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class NumLocalDofsLambda(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._k = space.abstract.k
        self._cache = {}
        self._freeze()

    def __call__(self, degree):
        """Making the local numbering for degree."""
        p = self._space[degree].p
        if p in self._cache:
            return self._cache[p]
        else:
            num_dofs = getattr(self, f"_k{self._k}")(p)
            self._cache[p] = num_dofs
            return num_dofs

    @staticmethod
    def _k0(p):
        """"""
        _q_ = (p+1) * (p+1)
        _t_ = (p+1) * p + 1
        return {
            'q': _q_,
            't': _t_,
        }

    @staticmethod
    def _k1(p):
        """"""
        Px = (p+1) * p
        Py = p * (p+1)
        _q_ = Px + Py

        Py = p * p
        _t_ = Px + Py

        return {
            'q': _q_,
            't': _t_,
        }

    @staticmethod
    def _k2(p):
        """"""
        _ = p * p
        return {
            'q': _,
            't': _,
        }
