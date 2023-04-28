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
from msepy.form.cochain.main import MsePyRootFormCochain
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
        self._numbering = None
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
    def m(self):
        return self.space.m  # esd

    @property
    def n(self):
        return self.space.n  # mesh.ndim

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
        """Continuous form (a shell, the real `cf` is in `cf.field`) of this root-form"""
        if self._cf is None:
            self._cf = MsePyContinuousForm(self)
        return self._cf

    @cf.setter
    def cf(self, cf):
        """Setter of `cf`.

        We actually set `cf.field`, use a shell `cf` to enabling extra checkers and so on.
        """
        self.cf.field = cf

    def set_cf(self, cf):
        """A more reasonable method name."""
        self.cf = cf

    @property
    def cochain(self):
        """The cochain class."""
        if self._cochain is None:
            self._cochain = MsePyRootFormCochain(self)
        return self._cochain


if __name__ == '__main__':
    # python msepy/form/main.py
    import numpy as np
    import __init__ as ph
    space_dim = 1
    ph.config.set_embedding_space_dim(space_dim)

    manifold = ph.manifold(space_dim)
    mesh = ph.mesh(manifold)
    L0 = ph.space.new('Lambda', 0)
    f0 = L0.make_form('0-f', 'f^0')
    ph.space.finite(5)

    msepy, obj = ph.fem.apply('msepy', locals())

    manifold = obj['manifold']
    mesh = obj['mesh']
    f0 = obj['f0']

    msepy.config(manifold)('crazy', c=0.3, periodic=False, bounds=[[0, 2] for _ in range(space_dim)])
    # msepy.config(mnf)('backward_step')
    msepy.config(mesh)([3 for _ in range(space_dim)])

    def fx(t, x):
        return np.sin(2*np.pi*x) + t
    scalar = ph.vc.scalar(fx)
    f0.cf = fx
