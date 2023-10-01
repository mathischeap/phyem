# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from msehy.py2.form.numeric.interp import MseHyPy2FormNumericInterp
from tools.dds.region_wise_structured import DDSRegionWiseStructured


class MseHyPy2FormNumeric(Frozen):
    """Numeric methods are approximate; less accurate than, for example, reconstruction."""

    def __init__(self, rf, t, g):
        """"""
        self._f = rf
        self._t = t
        self._g = g
        self._interp = None
        self._freeze()

    def region_wise_reconstruct(self, r, s, method='linear'):
        """

        Parameters
        ----------
        r :
            2d array, all entries in [0, 1] since the interpolation is mesh-region-wise.
        s :
            2d array, all entries in [0, 1] since the interpolation is mesh-region-wise.
        method : {'linear', }

        Returns
        -------

        """
        if self._interp is None:
            self._interp = MseHyPy2FormNumericInterp(self._f, self._t, self._g)
        else:
            pass

        interp = self._interp(method)   # mesh-region-wise interp functions.

        if not isinstance(r, np.ndarray):
            r = np.array(r)
        if not isinstance(s, np.ndarray):
            s = np.array(s)
        assert np.ndim(r) == np.ndim(s) == 2, f"r, s must be 2d."
        assert np.shape(r) == np.shape(s), f"r, s shape do not match."
        assert np.min(r) >= 0 and np.max(r) <= 1, f"all entries of r must be in [0,1]."
        assert np.min(s) >= 0 and np.max(s) <= 1, f"all entries of r must be in [0,1]."

        representative = self._f.mesh[self._g]
        background = representative.background
        regions = background.manifold.regions
        Xd = dict()
        Yd = dict()

        results = None
        for region in interp:
            itp_s = interp[region]
            if results is None:
                results = list()
                for _ in itp_s:
                    results.append(
                        dict()
                    )
            else:
                assert len(results) == len(itp_s)

        for region in regions:
            x, y = regions[region]._ct.mapping(r, s)
            Xd[region] = x
            Yd[region] = y
            itp_s = interp[region]
            for j, itp in enumerate(itp_s):
                results[j][region] = itp(x, y)

        return DDSRegionWiseStructured([Xd, Yd], results)
