# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""

from tools.frozen import Frozen


class MsePyContinuousForm(Frozen):
    """"""

    def __init__(self, rf):
        """"""
        self._f = rf
        self._field = None
        self._freeze()

    def __iter__(self):
        """iter over all region indices."""
        for i in self._f.mesh.regions:
            yield i

    def __getitem__(self, region_index):
        """get the cf for region #`region_index`."""
        return self.field[region_index]

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
