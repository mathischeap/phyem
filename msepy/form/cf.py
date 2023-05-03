# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""
import numpy as np
from tools.frozen import Frozen


class MsePyContinuousForm(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._field = None
        self._freeze()

    def __getitem__(self, t):
        """Return the partial functions at time `t` in all regions."""
        field_t = dict()
        for i in self._f.mesh.regions:
            field_t[i] = self.field[i][t]
        return MsePyContinuousFormPartialTime(self._f, field_t)

    @property
    def field(self):
        """the cf."""
        return self._field

    @field.setter
    def field(self, _field):
        """"""
        regions = self._f.mesh.regions
        if isinstance(_field, dict):
            self._field = _field
        else:
            _fd = dict()
            for i in regions:
                _fd[i] = _field
            self._field = _fd


class MsePyContinuousFormPartialTime(Frozen):
    """"""

    def __init__(self, rf, field_t):
        """"""
        self._f = rf
        self._field = field_t
        self._freeze()

    def __call__(self, *xyz, axis=0):
        """No matter what `axis` is, in the results, also `axis` is elements-wise.

        Parameters
        ----------
        xyz
        axis :
            The element-wise data is along this axis.

        Returns
        -------

        """
        values = None
        for ri in self._field:
            elements = self._f.mesh.elements._elements_in_region(ri)
            start, end = elements
            xyz_region_wise = list()
            for coo in xyz:
                xyz_region_wise.append(
                    coo.take(indices=range(start, end), axis=axis)
                )

            func = self._field[ri]

            value_region = func(*xyz_region_wise)
            num_components = len(value_region)

            if values is None:
                values = [[] for _ in range(num_components)]
            else:
                pass

            for c in range(num_components):
                values[c].append(value_region[c])

        for i, val in enumerate(values):
            values[i] = np.concatenate(val, axis=axis)

        return values
