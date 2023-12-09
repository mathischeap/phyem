# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import numpy as np


class ParticularCochainAtTimeInstant(Frozen):
    """"""

    def __init__(self, rf, t):
        """"""
        assert rf._is_base, f"rf must be a base root-form."
        self._f = rf
        self._t = t
        self._local_cochain = None
        self._local_cochain_caller = None
        self._type = None
        self._freeze()

    def __repr__(self):
        """"""
        my_repr = rf"<Cochain at time={self._t} of "
        rf_repr = self._f.__repr__()
        return my_repr + rf_repr + '>'

    def _receive(self, cochain):
        """"""
        # check what we kind of cochain we receive, and convert it to `local` type any way.
        if isinstance(cochain, dict):
            gm = self._f.cochain.gathering_matrix
            assert len(cochain) == len(gm), \
                f'cochain length {len(cochain)} does not match that of the gathering matrix.'
            for index in gm:
                assert index in cochain, f"cochain for element #{index} is missing"
                assert isinstance(cochain[index], np.ndarray), f"local cochain in element #{index} is not a ndarray."
                assert np.shape(cochain[index]) == (gm.num_local_dofs(index),), \
                    f"shape of local cochain in element #{index} is wrong"
            self._local_cochain = cochain
            self._type = 'dict'
        elif callable(cochain):
            self._local_cochain_caller = cochain
            self._type = 'realtime'
        else:
            raise Exception(f"Cannot receive cochain from {cochain.__class__}")
        assert self._type is not None, f"When receive a cochain, its type must be specified."

    @property
    def local(self):
        """The cochain provided in a local format."""
        if self._type == 'dict':
            return self._local_cochain
        elif self._type == 'realtime':
            local_cochain = self._local_cochain_caller(self._t)
            gm = self._f.cochain.gathering_matrix
            assert isinstance(local_cochain, dict) and len(local_cochain) == len(gm), \
                f'cochain length does not match that of the gathering matrix.'
            for index in gm:
                assert index in local_cochain, f"cochain for element #{index} is missing"
                assert isinstance(local_cochain[index], np.ndarray), \
                    f"local cochain in element #{index} is not a ndarray."
                assert np.shape(local_cochain[index]) == (gm.num_local_dofs(index),), \
                    f"shape of local cochain in element #{index} is wrong"

            return local_cochain
        else:
            raise Exception

    def of_dof(self, i, average=True):
        """The cochain for the global dof `#i`."""
        elements_local_indices = self._f.cochain.gathering_matrix._find_elements_and_local_indices_of_dofs(i)

        i = list(elements_local_indices.keys())[0]
        elements, local_rows = elements_local_indices[i]
        values = list()
        local = self.local
        for e, i in zip(elements, local_rows):
            values.append(
                local[e][i]
            )

        if average:
            return sum(values) / len(values)
        else:
            return values[0]
