# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen


class MseHyPy2LocalNumberingLambda(Frozen):
    """Generation in-dependent."""

    def __init__(self, space):
        """Generation in-dependent."""
        self._space = space
        self._k = space.abstract.k
        self._orientation = space.abstract.orientation
        self._freeze()

    def __call__(self, degree):
        """Generation in-dependent."""
        p = self._space[degree].p
        if self._k == 1:
            method_name = f"_k{self._k}_{self._orientation}"
        else:
            method_name = f"_k{self._k}"
        ln = getattr(self, method_name)(p)
        return ln

    @staticmethod
    def _k0(p):
        """"""
        px, py = p

        local_numbering = np.arange(0, (px+1) * (py+1)).reshape((px+1, py+1), order='F')
        _q_ = (local_numbering,)  # do not remove (,)

        assert px == py
        ln = np.zeros((px+1, py+1))
        ln[1:, :] = np.arange(1, 1+px*(py+1)).reshape((px, py+1), order='F')
        _t_ = (ln, )

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
        # segments perp to x-axis
        local_numbering_dy = np.arange(0, Px).reshape((px+1, py), order='F')
        # segments perp to y-axis
        local_numbering_dx = np.arange(Px, Px + Py).reshape((px, py+1), order='F')
        _q_ = local_numbering_dy, local_numbering_dx

        assert px == py
        LN_dy = np.arange(0, px*py).reshape((px, py), order='F')
        LN_dx = np.arange(px*py, px*py + px*(py+1)).reshape((px, py+1), order='F')
        _t_ = (LN_dy, LN_dx)
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
        local_numbering_dx = np.arange(0, Px).reshape((px, py+1), order='F')
        local_numbering_dy = np.arange(Px, Px + Py).reshape((px+1, py), order='F')
        _q_ = local_numbering_dx, local_numbering_dy

        assert px == py
        LN_dx = np.arange(0, px*(py+1)).reshape((px, py+1), order='F')
        LN_dy = np.arange(px*(py+1), px*(py+1) + px * py).reshape((px, py), order='F')
        _t_ = (LN_dx, LN_dy)

        return {
            'q': _q_,
            't': _t_,
        }

    @staticmethod
    def _k2(p):
        """"""
        px, py = p
        local_numbering = np.arange(0, px * py).reshape((px, py), order='F')
        _q_ = (local_numbering,)  # do not remove (,)

        return {
            'q': _q_,
            't': _q_,
        }
