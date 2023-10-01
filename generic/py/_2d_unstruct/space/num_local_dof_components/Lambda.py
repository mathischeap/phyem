# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class NumLocalDofComponentsLambda(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._k = space.abstract.k
        self._orientation = space.abstract.orientation
        self._freeze()

    def __call__(self, degree):
        """Making the local numbering for degree."""
        p = self._space[degree].p

        if self._k == 1:
            method_name = f"_k{self._k}_{self._orientation}"
        else:
            method_name = f"_k{self._k}"

        LN = getattr(self, method_name)(p)

        return LN

    @staticmethod
    def _k0(p):
        """"""
        _q_ = [(p+1) * (p+1), ]
        _t_ = [(p+1) * p + 1, ]
        return {
            'q': _q_,
            't': _t_,
        }

    @staticmethod
    def _k1_inner(p):
        """"""
        Px = p * (p+1)
        Py = (p+1) * p
        _q_ = [Px, Py]

        Py = p * p
        _t_ = [Px, Py]

        return {
            'q': _q_,
            't': _t_,
        }

    @staticmethod
    def _k1_outer(p):
        """"""
        Px = (p+1) * p
        Py = p * (p+1)
        _q_ = Px, Py

        Px = p * p
        _t_ = [Px, Py]

        return {
            'q': _q_,
            't': _t_,
        }

    @staticmethod
    def _k2(p):
        """"""
        _ = [p * p, ]

        return {
            'q': _,
            't': _,
        }
