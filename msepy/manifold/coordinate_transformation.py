# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
created at: 3/31/2023 2:35 PM
"""
import sys

if './' not in sys.path:
    sys.path.append('./')
from src.tools.frozen import Frozen


class MsePyManifoldsCoordinateTransformation(Frozen):
    """"""
    def __init__(self, mf):
        assert mf.regions is not None, f"pls set regions for manifold {mf} before accessing ct."
        self._mf = mf
        self._freeze()

    def __repr__(self):
        """"""
        return f"<CT of " + self._mf.__repr__() + ">"

    def mapping(self, *rst, regions=None):
        """"""
        if regions is None:
            regions = range(len(self._mf.regions))
        else:
            if isinstance(regions, int):
                regions = [regions, ]
            else:
                pass

        _2return = dict()

        for ri in regions:
            assert ri in self._mf.regions, f"region index = {ri} is not a valid one!"
            _2return[ri] = self._mf.regions[ri]._ct.mapping(*rst)

        return _2return

    def Jacobian_matrix(self, *rst, regions=None):
        """"""
        if isinstance(regions, int):
            assert regions in self._mf.regions, f"region index = {regions} is not a valid one!"
            rct = self._mf.regions[regions]._ct
            xyz = rct.Jacobian_matrix(*rst)
            return {regions: xyz}  # just to make the output consistent.

        else:
            if regions is None:
                regions = range(len(self._mf.regions))
            else:
                raise Exception()

        cache_rmt = dict()  # a cache whose keys are region mtypes.

        _2return = dict()

        for ri in regions:
            assert ri in self._mf.regions, f"region index = {ri} is not a valid one!"

            rct = self._mf.regions[ri]._ct
            rct_signature = rct.mtype.signature

            if rct_signature in cache_rmt:
                pass
            else:
                xyz = rct.Jacobian_matrix(*rst)
                cache_rmt[rct_signature] = xyz

            _2return[ri] = cache_rmt[rct_signature]

        return _2return
