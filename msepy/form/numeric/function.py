# -*- coding: utf-8 -*-
"""
"""
from tools.frozen import Frozen
from tools.functions.time_space._2d.wrappers.scalar import T2dScalar


class MsePyRootFormNumericFunction(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._freeze()

    def local_energy_with_time_shift(self, time_shift=True):
        """"""
        def ture_local_energy_computer(t, *xyz):
            return self._local_energy_computer(t, *xyz, time_shift=time_shift)
        return T2dScalar(ture_local_energy_computer)

    def _local_energy_computer(self, t, *xyz, time_shift=False):
        """Compute 0.5 * u.dot(u) where u is the form at coordinates *xyz and time t.

        Parameters
        ----------
        t
        xyz
        time_shift : {bool, int, float}

        """
        if time_shift is True:  # auto shift to the newest time no matter what t is.
            t = self._f.cochain.newest
        elif time_shift is False:
            pass
        elif isinstance(time_shift, (int, float)):
            t += time_shift
        else:
            raise Exception()

        indicator = self._f.space.abstract.indicator
        if indicator == 'Lambda':
            space = self._f.space.abstract
            m, n, k = space.m, space.n, space.k
            if m == n == 2 and k in (0, 2):
                shape = [1]   # scalar in 2d
                ndim = 2      # scalar in 2d
            elif m == n == 2 and k == 1:
                shape = [2]   # vector in 2d
                ndim = 2      # vector in 2d
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        final_itp = self._f.numeric.interp(t=t)

        if shape == [2] and ndim == 2:  # vector in 2d
            itp0, itp1 = final_itp
            u = itp0(*xyz)
            v = itp1(*xyz)
            energy = 0.5 * (u**2 + v**2)
        else:
            raise NotImplementedError()

        return energy
