# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
created at: 3/16/2023 5:29 PM
"""

from tools.frozen import Frozen
import numpy as np
from msepy.tools.vector.local import MsePyLocalVector
from msepy.tools.vector.assembled import MsePyAssembledVector


class _CochainAtOneTime(Frozen):
    """"""

    def __init__(self, rf, t):
        """"""
        assert rf._is_base, f"rf must be a base root-form."
        self._f = rf
        self._t = t
        self._local_cochain = None
        self._freeze()

    def __repr__(self):
        """"""
        rf_repr = self._f.__repr__()
        my_repr = rf"<Cochain at time={self._t} of "
        super_repr = super().__repr__().split(' object')[1]
        return my_repr + rf_repr + super_repr

    def _receive(self, cochain):
        """"""
        # check what we kind of cochain we receive, and convert it to `local` type any way.
        if cochain.__class__.__name__ == 'ndarray' and np.ndim(cochain) == 2:
            # TODO check shape with gathering_matrix
            self._local_cochain = cochain
        elif cochain.__class__ is MsePyLocalVector:
            raise NotImplementedError()
        elif cochain.__class__ is MsePyAssembledVector:
            raise NotImplementedError()
        else:
            raise Exception(f"Cannot receive cochain from {cochain}")

    @property
    def local(self):
        """2d-numpy-array."""
        return self._local_cochain
