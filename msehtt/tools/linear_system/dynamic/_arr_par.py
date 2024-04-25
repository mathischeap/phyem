# -*- coding: utf-8 -*-
r"""
"""

from tools.frozen import Frozen
from src.spaces.main import _sep

from src.config import _form_evaluate_at_repr_setting
_rf_evaluate_at_lin_repr = _form_evaluate_at_repr_setting['lin']


_locals = locals()

_indicator_templates = {}

_setting_ = {
    'base': dict()
}


def _indicator_check():
    for i, key in enumerate(_locals):
        if key[:12] == '_VarSetting_':
            default_setting = _locals[key]
            splits = default_setting[1].split(_sep)
            indicator = splits[0]
            key_words = splits[1:]
            symbol = default_setting[0]
            assert indicator not in _indicator_templates, 'repeated indicator found.'
            _indicator_templates[indicator] = {
                'key_words': key_words,
                'symbol': symbol
            }


def _parse_root_form(root_form_vec_lin_repr):
    """"""
    forms = _setting_['base']['forms']
    rf = None   # msepy rf

    for rf_pure_lin_repr in forms:
        if rf_pure_lin_repr == root_form_vec_lin_repr:
            rf = forms[rf_pure_lin_repr]
        else:
            pass

    assert rf is not None, f"DO NOT find a msepy root-form, something is wrong."

    if _rf_evaluate_at_lin_repr in rf.abstract._pure_lin_repr:
        assert rf._pAti_form['base_form'] is not None, f"must be a particular root-form!"
    else:  # it is a general (not for a specific time step for example) vector of the root-form.
        assert rf._pAti_form['base_form'] is None, f"must be a general root-form!"

    dynamic_cochain_vec = rf.cochain.dynamic_vec
    return dynamic_cochain_vec, rf.abstract.ap()._sym_repr, rf.cochain._ati_time_caller
