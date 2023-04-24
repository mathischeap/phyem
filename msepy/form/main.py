# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 1:22 PM on 4/17/2023
"""

import sys
from typing import Dict

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.form.cf import MsePyContinuousForm
from msepy.form.cochain import MsePyRootFormCochain
from msepy.form.realtime import MsePyRootFormRealTimeCopy


class MsePyRootForm(Frozen):
    """"""

    def __init__(self, abstract_root_form):
        """"""
        self._abstract = abstract_root_form
        abstract_root_form._objective = self
        abstract_space = abstract_root_form.space
        self._space = abstract_space._objective
        degree = self.abstract._degree
        assert degree is not None, f"{abstract_root_form} has no degree."
        self._degree = None
        self.space.finite.new(degree)
        self.space.finite.specify_form(self, degree)
        self._cf = None
        self._cochain = None  # do not initialize cochain here!
        self._pAti_form: Dict = {
            'base_form': None,
            'ats': None,
            'ati': None
        }
        self._ats_particular_forms = dict()   # the abstract forms based on this form.
        self._freeze()

    @property
    def abstract(self):
        """the abstract object this root-form is for."""
        return self._abstract

    def __repr__(self):
        """repr"""
        ab_rf_repr = self._abstract.__repr__().split(' at ')[0][1:]
        return "<MsePy " + ab_rf_repr + super().__repr__().split(" object")[1]

    def __getitem__(self, t):
        """return the realtime copy of `self` at time `t`."""
        if self._is_base():
            return MsePyRootFormRealTimeCopy(self, t)
        else:
            return MsePyRootFormRealTimeCopy(self._base, t)

    def _is_base(self):
        """Am I a base root-form (not abstracted at a time.)"""
        return self._base is None

    @property
    def _base(self):
        """The base root-form I have."""
        return self._pAti_form['base_form']

    @property
    def space(self):
        """The `MsePySpace` I am in."""
        return self._space

    @property
    def mesh(self):
        """The objective mesh."""
        return self.space.mesh

    @property
    def degree(self):
        """The degree of my space."""
        return self._degree

    @property
    def cf(self):
        """Continuous form of this root-form"""
        if self._cf is None:
            self._cf = MsePyContinuousForm(self)
        return self._cf

    @cf.setter
    def cf(self, cf):
        """Setter of `cf`."""
        self.cf.field = cf

    @property
    def cochain(self):
        """The cochain class."""
        if self._cochain is None:
            self._cochain = MsePyRootFormCochain(self)
        return self._cochain


if __name__ == '__main__':
    # python msepy/form/main.py
    pass
