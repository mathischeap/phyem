# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from tools.functions.time_space._2d.wrappers.scalar import T2dScalar


class MsePyRootFormNumericTimeSpaceProperty(Frozen):
    """Like the TimeSpaceFunction, but here we return for example scalar instances."""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._freeze()

    def L2_energy(self, t=None):
        """ 0.5 * (u dot u) where u represents the form, i.e. `self._f`.
        """
        if t is None:
            # return a scalar instance taking (t, *xyz) as inputs.
            # But no matter what t is given, it will compute the
            # local energy at the newest cochain time.
            def ture_local_energy_computer(t, *xyz):
                return self._f.numeric.tsf._local_energy_computer(t, *xyz, time_shift=True)

            return T2dScalar(ture_local_energy_computer)
        else:
            raise NotImplementedError()
