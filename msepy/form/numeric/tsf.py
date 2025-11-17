# -*- coding: utf-8 -*-
r"""
"""
from numpy import isnan

from phyem.tools.frozen import Frozen


class MsePyRootFormNumericTimeSpaceFunction(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._freeze()

    def __call__(self):
        """Return the form as a numerical function copy which takes (t, *xyz) as inputs. For example,
        Suppose there are two scalar valued 1-forms, A and B,

            a = A.numeric.function(None)
            b = B.numeric.function(None)

        We got `a` and `b` which are two vectors. Then we can do for example

            c = a.dot(b)
            d = a.cross_product(b)

        and so on.
        """
        raise NotImplementedError()

    def L2_energy(self, t, *xyz):
        """0.5 * (u dot u) where u is the form, i.e. `self._f`. """
        return self._local_energy_computer(t, *xyz)

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
                dtype = '2d scalar'
            elif m == n == 2 and k == 1:
                dtype = '2d vector'
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

        final_itp = self._f.numeric.interp(t=t)

        if dtype == '2d vector':  # vector in 2d
            itp0, itp1 = final_itp
            u = itp0(*xyz)
            v = itp1(*xyz)
            energy = 0.5 * (u**2 + v**2)

        else:
            raise NotImplementedError()

        if (isnan(energy)).any():
            raise Exception

        return energy
