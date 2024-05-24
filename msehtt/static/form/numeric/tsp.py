# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from tools.functions.time_space._2d.wrappers.scalar import T2dScalar


class MseHtt_Form_Numeric_TimeSpaceProperties(Frozen):
    """"""

    def __init__(self, f):
        """"""
        self._f = f
        self._freeze()

    def L2_energy(self, t=None):
        """Return a function e(t, x, y) gives the L2-energy, i.e. 0.5 * (f, f), at (t, x, y).

        The time of f is ``t``. When ``t`` is None, we always use the newest time of its cochain. So this energy
        will update automatically and the t in e(t, x, y) does not have any effect.

        """
        if t is None:
            ndim = self._f.space.n
            if ndim == 2:
                return T2dScalar(self.___newest_energy___)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def ___newest_energy___(self, t, *xyz):
        """"""
        _ = t  # t has no effect since we will always use the newest t of the cochain.
        dtype, V = self._f.numeric._interpolate_()

        if dtype == '2d-scalar':
            return 0.5 * V[0](*xyz) ** 2

        elif dtype == '2d-vector':
            return 0.5 * (V[0](*xyz) ** 2 + V[1](*xyz) ** 2)

        else:
            raise NotImplementedError()
