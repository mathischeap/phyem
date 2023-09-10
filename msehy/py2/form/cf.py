# -*- coding: utf-8 -*-
r"""
"""
from msepy.form.cf import MsePyContinuousForm
from tools.functions.region_wise_wrapper import RegionWiseFunctionWrapper


class MseHyPy2ContinuousForm(MsePyContinuousForm):
    """"""

    # noinspection PyMissingConstructor
    def __init__(self, rf):
        """"""
        self._f = rf
        self._regions = self._f.mesh.background.regions
        self._field = None
        self._shape = None
        self._freeze()

    def __getitem__(self, t):
        """Return the partial functions at time `t` in all regions."""
        field_t = dict()
        for i in self._regions:
            field_t[i] = self.field[i][t]
        return RegionWiseFunctionWrapper(field_t)
