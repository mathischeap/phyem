# -*- coding: utf-8 -*-
r"""
"""
from msehtt.tools.vector.static.local import MseHttStaticLocalVector


class MseHttStaticCochainVector(MseHttStaticLocalVector):
    """"""

    def __init__(self, rf, t, _2d_data, gathering_matrix):
        """"""
        self._f = rf
        self._time = t
        super().__init__(_2d_data, gathering_matrix)
        self._freeze()

    def override(self):
        """override `self._data` to be the cochain of `self._f` at time `self._t`."""

        assert self._dtype != 'None', f"I have no data."
        self._f[self._time].cochain = self.data_dict

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f"<static-cochain-vector of {self._f.abstract._sym_repr} @ {self._time}" + super_repr
