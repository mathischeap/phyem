# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 1:22 PM on 4/17/2023
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen


class MsePyRootForm(Frozen):
    """"""

    def __init__(self, abstract_root_form):
        """"""
        self._abstract = abstract_root_form
        abstract_space = abstract_root_form.space
        self._space = abstract_space._objective
        degree = self.abstract._degree
        assert degree is not None, f"{abstract_root_form} has no degree."
        self._degree = None
        self.space.finite.new(degree)
        self.space.finite.specify_form(self, degree)
        self._freeze()

    @property
    def abstract(self):
        """the abstract object this root-form is for."""
        return self._abstract

    def __repr__(self):
        """repr"""
        ab_rf_repr = self._abstract.__repr__().split(' at ')[0][1:]
        return "<MsePy " + ab_rf_repr + super().__repr__().split(" object")[1]

    @property
    def space(self):
        """The `MsePySpace` I am in."""
        return self._space

    @property
    def degree(self):
        """The degree of my space."""
        return self._degree


if __name__ == '__main__':
    # python msepy/form/main.py
    pass
