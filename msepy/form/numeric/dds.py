# -*- coding: utf-8 -*-
"""Discrete Data Structure.
"""
from tools.frozen import Frozen
import numpy as np
from tools.dds.region_wise_structured import DDSRegionWiseStructured


class MsePyRootFormNumericDiscreteDataStructure(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._freeze()

    def ___parse_t___(self, t):
        if t is None:
            t = self._f.cochain.newest
        else:
            pass
        return t

    def ___decide_type___(self):
        """"""
        indicator = self._f.space.abstract.indicator
        if indicator == 'Lambda':
            space = self._f.space.abstract
            m, n, k = space.m, space.n, space.k
            if m == n == 2 and k in (0, 2):
                return 2, 'scalar'
            elif m == n == 2 and k == 1:
                return 2, 'vector'
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def __call__(self, **kwargs):
        """Make the default dds."""
        return self.rws(**kwargs)

    def rws(self, t=None, ddf=5):
        """Region-wise structured dds of the form at time `t`.

        Parameters
        ----------
        t
        ddf :
            Data density factor.

        """
        t = self.___parse_t___(t)
        ndim, dtype = self.___decide_type___()
        linspace = np.linspace(-1, 1, ddf)
        grid_xi_et_sg = [linspace for _ in range(ndim)]
        xyz, value = self._f.reconstruct(t, *grid_xi_et_sg)

        if (ndim, dtype) == (2, 'scalar'):
            x, y = xyz
            v = value[0]
            x, y, v = self._f.mesh._regionwsie_stack(x, y, v)
            return DDSRegionWiseStructured([x, y], [v])

        elif (ndim, dtype) == (2, 'vector'):
            x, y = xyz
            u, v = value
            x, y, u, v = self._f.mesh._regionwsie_stack(x, y, u, v)
            return DDSRegionWiseStructured([x, y], [u, v])

        else:
            raise NotImplementedError()
