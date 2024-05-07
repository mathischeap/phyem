# -*- coding: utf-8 -*-
r"""
"""
from src.spaces.main import _str_degree_parser
from src.spaces.main import _sep
from src.config import _form_evaluate_at_repr_setting
_rf_evaluate_at_lin_repr = _form_evaluate_at_repr_setting['lin']

from msehtt.tools.matrix.static.local import MseHttStaticLocalMatrix


# noinspection PyUnresolvedReferences
from src.spaces.main import *
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


_indicator_check()

_indicator_cache = {}


def _find_indicator(default_setting):
    key = str(default_setting)
    if key in _indicator_cache:
        pass
    else:
        _indicator_cache[key] = default_setting[1].split(_sep)[0]
    return _indicator_cache[key]


def _base_spaces():
    return _setting_['base']['spaces']


def _base_forms():
    return _setting_['base']['forms']


def _find_space_through_pure_lin_repr(_target_space_lin_repr):
    """"""
    spaces = _base_spaces()
    the_msepy_space = None
    for space_lin_repr in spaces:
        msepy_space = spaces[space_lin_repr]
        abs_space_pure_lin_repr = msepy_space.abstract._pure_lin_repr
        if abs_space_pure_lin_repr == _target_space_lin_repr:
            the_msepy_space = msepy_space
            break
        else:
            pass
    assert the_msepy_space is not None, f"Find no msepy space."
    return the_msepy_space


def _parse_root_form(root_form_vec_lin_repr):
    """"""
    forms = _base_forms()
    rf = None   # the root-form

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


def Parse__M_matrix(space, degree0, degree1):
    """"""
    degree0 = _str_degree_parser(degree0)
    degree1 = _str_degree_parser(degree1)
    space = _find_space_through_pure_lin_repr(space)
    if degree0 == degree1:
        degree = degree0
        gm = space.gathering_matrix(degree)
        m, cache_key_dict = space.mass_matrix(degree)
        M = MseHttStaticLocalMatrix(  # make a new copy every single time.
            m,
            gm,
            gm,
            cache_key=cache_key_dict
        )
        return M, None  # time_indicator is None, mean M is same at all time.
    else:
        raise NotImplementedError()


def Parse__E_matrix(space, degree):
    """"""
    degree = _str_degree_parser(degree)
    msehtt_space = _find_space_through_pure_lin_repr(space)
    gm0 = msehtt_space.gathering_matrix._next(degree)
    gm1 = msehtt_space.gathering_matrix(degree)
    e, cache_key_dict = msehtt_space.incidence_matrix(degree)
    E = MseHttStaticLocalMatrix(  # make a new copy every single time.
        e,
        gm0,
        gm1,
        cache_key=cache_key_dict
    )
    return E, None  # time_indicator is None, mean E is same at all time.
