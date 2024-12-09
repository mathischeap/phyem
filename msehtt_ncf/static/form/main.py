# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from typing import Dict
from src.form.main import Form

from msehtt.static.form.cochain.main import MseHttCochain


class MseHtt_NCF_Static_Form(Frozen):
    """"""

    def __init__(self, abstract_root_form):
        """"""
        self._pAti_form: Dict = {
            'base_form': None,  # the base form
            'ats': None,  # abstract time sequence
            'ati': None,  # abstract time instant
        }
        self._ats_particular_forms = dict()   # the abstract forms based on this form.
        assert abstract_root_form.__class__ is Form, f"I need an abstract form."
        self._degree = abstract_root_form._degree
        self._abstract = abstract_root_form

        self._name_ = None  # name of this form

        self._tgm = None  # the msehtt great mesh
        self._tpm = None  # the msehtt partial mesh
        self._space = None  # the msehtt space
        self._manifold = None   # the msehtt manifold

        self._cf = None
        self._cochain = None
        self._im = None
        self._bi = None
        self._numeric = None
        self._export = None
        self._project = None
        self._freeze()

    @property
    def abstract(self):
        """Return the abstract form."""
        return self._abstract

    def __repr__(self):
        """repr"""
        ab_rf_repr = self._abstract.__repr__().split(' at ')[0][1:]
        return "<MseHtt " + ab_rf_repr + super().__repr__().split(" object")[1]

    @property
    def degree(self):
        r"""The degree of the form."""
        return self._degree

    def _is_base(self):
        r"""Am I a base root-form (not abstracted at a time)?"""
        return self._base is None

    @property
    def _base(self):
        r"""The base root-form I have.

        if `self._is_base()`, return None. Else return the base form.
        """
        return self._pAti_form['base_form']

    @property
    def tgm(self):
        r"""Return the msehtt great mesh."""
        return self._tgm

    @property
    def tpm(self):
        r"""Return the msehtt partial mesh."""
        return self._tpm

    @property
    def space(self):
        r"""Return the msehtt space."""
        return self._space

    @property
    def manifold(self):
        r"""Return the msehtt manifold."""
        return self._manifold

    @property
    def name(self):
        r""""""
        if self._is_base():
            if self._name_ is None:
                self._name_ = 'msehtt-ncf-form: ' + self.abstract._sym_repr + ' = ' + self.abstract._lin_repr
            return self._name_
        else:
            if self._name_ is None:
                base_name = self._base.name
                ati = self._pAti_form['ati']
                self._name_ = base_name + '@' + ati.__repr__()
            return self._name_

    @property
    def cochain(self):
        """The cochain class."""
        if self._cochain is None:
            self._cochain = MseHttCochain(self)
        return self._cochain
