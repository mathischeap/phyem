# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import numpy as np
from msepy.tools.vector.static.local import MsePyStaticLocalVector
from msepy.tools.vector.static.assembled import MsePyStaticAssembledVector


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
        my_repr = rf"<Cochain at time={self._t} of "
        rf_repr = self._f.__repr__()
        super_repr = super().__repr__().split(' object')[1]
        return my_repr + rf_repr + super_repr

    def _receive(self, cochain):
        """"""
        # check what we kind of cochain we receive, and convert it to `local` type any way.
        if cochain.__class__.__name__ == 'ndarray' and np.ndim(cochain) == 2:
            gm = self._f.cochain.gathering_matrix
            assert np.shape(cochain) == gm.shape, f"local cochain shape = {np.shape(cochain)} wrong, " \
                                                  f"should be {gm.shape}."
            self._local_cochain = cochain
        elif cochain.__class__ is MsePyStaticLocalVector:
            raise NotImplementedError()
        elif cochain.__class__ is MsePyStaticAssembledVector:
            raise NotImplementedError()
        else:
            raise Exception(f"Cannot receive cochain from {cochain.__class__}")

    @property
    def local(self):
        """2d-numpy-array."""
        return self._local_cochain

    def of_dof(self, i, average=True):
        """The cochain for the global dof `#i`."""
        elements_local_indices = self._f.cochain.gathering_matrix._find_elements_and_local_indices_of_dofs(i)
        i = list(elements_local_indices.keys())[0]
        elements, local_rows = elements_local_indices[i]
        values = list()
        for e, i in zip(elements, local_rows):
            values.append(
                self.local[e][i]
            )
        if average:
            return sum(values) / len(values)
        else:
            return values[0]
