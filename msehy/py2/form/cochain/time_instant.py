# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

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
        self._g = generation
        assert generation >= 0, f"safety check in case we save -1 generation."
        self._freeze()

    @property
    def generation(self):
        """this cochain lives on this generation of the mesh."""
        return self._g

    def __repr__(self):
        """"""
        my_repr = rf"<Irregular cochain at time={self._t} of G[{self.generation}] "
        rf_repr = self._f.__repr__()
        super_repr = super().__repr__().split(' object')[1]
        return my_repr + rf_repr + super_repr

    def _receive(self, cochain):
        """"""
        num_local_dofs = self._f.space.num_local_dofs(self._f.degree)
        if isinstance(cochain, dict):
            # a dict whose keys are indices of all fundamental cell indices on self.generation.
            elements = self._f.mesh[self.generation]
            assert len(elements) == len(cochain), f"cochain length wrong!"
            for i in elements:
                assert i in cochain, f"We miss cochain for fc {i}."
                fc = elements[i]
                assert isinstance(cochain[i], np.ndarray) and cochain[i].shape == (num_local_dofs[fc._type],), \
                    f"cochain shape wrong for fc {i}, need {(num_local_dofs[fc._type],)}, get {cochain[i].shape}"
            self._local_cochain = cochain
            self._type = 'ndarray'
        else:
            raise NotImplementedError(f"cannot receive cochain of type {cochain.__class__}")

    @property
    def local(self):
        if self._type == 'ndarray':
            return self._local_cochain
        else:
            raise NotImplementedError(f"not implemented.")
