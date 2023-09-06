# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class _IrregularCochainAtOneTime(Frozen):
    """"""
    def __init__(self, rf, t, generation):
        """"""
        assert rf._is_base, f"rf must be a base root-form."
        self._f = rf
        self._t = t
        self._local_cochain = None
        self._local_cochain_caller = None
        self._type = None
        self._generation = generation
        self._freeze()

    @property
    def generation(self):
        """this cochain lives on this generation of the mesh."""
        return self._generation

    def __repr__(self):
        """"""
        my_repr = rf"<Irregular cochain at time={self._t} of G[{self.generation}] "
        rf_repr = self._f.__repr__()
        super_repr = super().__repr__().split(' object')[1]
        return my_repr + rf_repr + super_repr

    def _receive(self, cochain):
        """"""
