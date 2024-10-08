# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from importlib import import_module


class MseHtt_Static_Form_Project(Frozen):
    """"""

    def __init__(self, f, t):
        """"""
        self._f = f
        self._t = t
        self._freeze()

    def to(self, to_what_indicator, **kwargs):
        """"""
        space_indicator = self._f.space.indicator
        form_indicator = self._f.space.str_indicator
        path = self.__repr__().split('main.')[0][1:] + f"{space_indicator}" + f".{form_indicator}"
        module = import_module(path)
        to_form_cochain = getattr(module, 'to__' + to_what_indicator)(self._f, self._t, **kwargs)
        return to_form_cochain
