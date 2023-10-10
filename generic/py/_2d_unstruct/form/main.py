# -*- coding: utf-8 -*-
r"""
"""
from typing import Dict

from tools.frozen import Frozen

from generic.py._2d_unstruct.space.main import GenericUnstructuredSpace2D
from generic.py.cochain.main import Cochain
from generic.py.static import StaticCopy
from generic.py._2d_unstruct.form.cf import _2d_CF
from generic.py._2d_unstruct.form.boundary_integrate.main import Boundary_Integrate
from generic.py._2d_unstruct.form.visualize import GenericUnstructuredForm2D_Visualize


class GenericUnstructuredForm2D(Frozen):
    """"""

    def __init__(
            self,
            space, degree,
            dual_representation=False,
            base_form=None, ats=None, ati=None
    ):
        """"""
        assert space.__class__ is GenericUnstructuredSpace2D, f'Must be'
        self._space = space
        assert degree is not None, f'must give me a degree'
        self._degree = degree

        self._dual_representation = dual_representation

        self._pAti_form: Dict = {
            'base_form': base_form,   # the base form
            'ats': ats,   # abstract time sequence
            'ati': ati,   # abstract time instant
        }
        self._parse_base = True   # parse the base form for the first time.
        self.___base___ = None

        self._cochain = None
        self._cf = None
        self._bi = None
        self._vis = None

        self._name = rf'form@${self.space.abstract._sym_repr}$'
        self._freeze()

    def is_dual_representation(self):
        """When it is a dual representation, we should process the dofs."""
        return self._dual_representation

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return f"<form in {self._space}" + super_repr

    @property
    def space(self):
        """"""
        return self._space

    @property
    def mesh(self):
        """"""
        return self.space.mesh

    @property
    def name(self):
        """"""
        return self._name

    @property
    def degree(self):
        """"""
        return self._degree

    def _is_base(self):
        """Am I a base root-form (not abstracted at a time.)"""
        return self._base is None

    @property
    def _base(self):
        """The base root-form I have."""
        if self._parse_base:
            raw_base = self._pAti_form['base_form']
            if raw_base is None:
                self.___base___ = None
            else:
                if hasattr(raw_base, 'generic'):
                    generic = raw_base.generic
                    self._pAti_form['base_form'] = generic  # renew the base
                    self.___base___ = generic
                else:
                    self.___base___ = raw_base
            self._parse_base = False
        return self.___base___

    @property
    def cochain(self):
        """"""
        if self._cochain is None:
            self._cochain = Cochain(self)
        return self._cochain

    @property
    def cf(self):
        """"""
        return self._cf

    @cf.setter
    def cf(self, _cf):
        """"""
        self._cf = _2d_CF(self, _cf)

    def __getitem__(self, t):
        """Return the static copy of self at time `t`."""
        if t is None:
            t = self.cochain.newest
            assert t is not None, f'I have no newest cochain.'
        else:
            pass
        t = self.cochain._parse_t(t)  # round off the truncation error to make it clear.
        if isinstance(t, (int, float)):
            if self._is_base():
                return StaticCopy(self, t)
            else:
                return StaticCopy(self._base, t)
        else:
            raise Exception(f"cannot accept t={t}.")

    def reduce(self, t, update_cochain=True, target=None):
        """"""
        if target is None:
            cochain_local = self.space.reduce(self.cf, t, self.degree)
        else:
            cochain_local = self.space.reduce(target, t, self.degree)

        if update_cochain:
            self[t].cochain = cochain_local
        else:
            pass

        return cochain_local

    def reconstruct(self, t, xi, et, ravel=False, element_range=None):
        """"""
        cochain = self.cochain[t]
        return self._space.reconstruct(cochain, xi, et, ravel,  element_range=element_range)

    @property
    def incidence_matrix(self):
        """"""
        return self._space.incidence_matrix(self.degree)

    @property
    def mass_matrix(self):
        """"""
        return self._space.mass_matrix(self.degree)

    def reconstruction_matrix(self, xi, et, element_range=None):
        """"""
        return self._space.reconstruction_matrix(self.degree, xi, et, element_range=element_range)

    @property
    def boundary_integrate(self):
        """"""
        if self._bi is None:
            self._bi = Boundary_Integrate(self)
        return self._bi

    @property
    def visualize(self):
        """visualize"""
        if self._vis is None:
            self._vis = GenericUnstructuredForm2D_Visualize(self)
        return self._vis
