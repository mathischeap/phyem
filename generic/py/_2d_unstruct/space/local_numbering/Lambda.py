# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import numpy as np


class LocalNumberingLambda(Frozen):
    """"""

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

        if p in self._cache:
            return self._cache[p]
        else:

            if self._k == 1:
                method_name = f"_k{self._k}_{self._orientation}"
            else:
                method_name = f"_k{self._k}"

            LN = getattr(self, method_name)(p)

            self._cache[p] = LN

            return LN

    @staticmethod
    def _k0(p):
        """"""
        local_numbering = np.arange(0, (p+1) * (p+1)).reshape((p+1, p+1), order='F')
        _q_ = (local_numbering,)  # do not remove (,)

        ln = np.zeros((p+1, p+1), dtype=int)
        ln[1:, :] = np.arange(1, 1+p*(p+1)).reshape((p, p+1), order='F')
        _t_ = (ln, )

        return {
            'q': _q_,
            't': _t_,
        }

    @staticmethod
    def _k1_outer(p):
        """"""
        Px = (p+1) * p
        Py = p * (p+1)
        # segments perp to x-axis
        local_numbering_dy = np.arange(0, Px).reshape((p+1, p), order='F')
        # segments perp to y-axis
        local_numbering_dx = np.arange(Px, Px + Py).reshape((p, p+1), order='F')
        _q_ = local_numbering_dy, local_numbering_dx

        P = p * p
        LN_dy = np.arange(0, P).reshape((p, p), order='F')
        LN_dx = np.arange(P, P + p*(p+1)).reshape((p, p+1), order='F')
        _t_ = (LN_dy, LN_dx)

        return {
            'q': _q_,
            't': _t_,
        }

    @staticmethod
    def _k1_inner(p):
        """"""
        Px = p * (p+1)
        Py = (p+1) * p
        local_numbering_dx = np.arange(0, Px).reshape((p, p+1), order='F')
        local_numbering_dy = np.arange(Px, Px + Py).reshape((p+1, p), order='F')
        _q_ = local_numbering_dx, local_numbering_dy

        _ = p*(p+1)
        LN_dx = np.arange(0, _).reshape((p, p+1), order='F')
        LN_dy = np.arange(_, _ + p * p).reshape((p, p), order='F')
        _t_ = (LN_dx, LN_dy)

        return {
            'q': _q_,
            't': _t_,
        }

    @staticmethod
    def _k2(p):
        """"""
        local_numbering = np.arange(0, p * p).reshape((p, p), order='F')
        _q_ = (local_numbering,)  # do not remove (,)

        return {
            'q': _q_,
            't': _q_,
        }
