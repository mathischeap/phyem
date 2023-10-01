# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from msepy.tools.vector.static.local import MsePyStaticLocalVector


class MsePyRootFormStaticCochainVector(MsePyStaticLocalVector):
    """"""

    def __init__(self, rf, t, _2d_data, gathering_matrix):
        """"""
        if _2d_data is None:
            pass
        else:
            assert isinstance(_2d_data, np.ndarray) and _2d_data.ndim == 2, \
                f"{MsePyRootFormStaticCochainVector} only accepts 2d array"
        self._f = rf
        self._time = t
        super().__init__(_2d_data, gathering_matrix)
        self._freeze()

    def override(self):
        """override `self._data` to be the cochain of `self._f` at time `self._t`."""
        if len(self.adjust) == 0 and len(self.customize) == 0:
            assert self.data is not None, f"I have no data."
            self._f[self._time].cochain = self.data
        else:
            raise NotImplementedError()
