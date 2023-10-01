# -*- coding: utf-8 -*-
r"""
"""
from src.spaces.main import _sep
from msehy.py2.main import base as base2

from src.config import get_embedding_space_dim
space_dim = get_embedding_space_dim()

if space_dim == 2:
    base = base2
else:
    raise NotImplementedError()

from src.spaces.main import _str_degree_parser
from src.spaces.main import *

from src.form.main import _global_root_forms_lin_dict

from src.config import _form_evaluate_at_repr_setting
_rf_evaluate_at_lin_repr = _form_evaluate_at_repr_setting['lin']


from msehy.py.operations.nonlinear.AxB_ip_C import _AxBipC


__all__ = [
    '_indicator_check',
    '_indicator_templates',
    '_find_indicator',
    '_find_from_bracket_ABC',
    '_find_space_through_pure_lin_repr',

    '_parse_root_form',

    'Parse__M_matrix',
    'Parse__E_matrix',

    'Parse__astA_x_astB_ip_tC',
    'Parse__astA_x_B_ip_tC',
    'Parse__A_x_astB_ip_tC',

]


_locals = locals()

_indicator_templates = {}


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
        else:
            pass


_indicator_cache = {}


def _find_indicator(default_setting):
    key = str(default_setting)
    if key in _indicator_cache:
        pass
    else:
        _indicator_cache[key] = default_setting[1].split(_sep)[0]
    return _indicator_cache[key]


def _find_from_bracket_ABC(default_repr, *ABC, key_words=("{A}", "{B}", "{C}")):
    """"""
    lin_reprs = default_repr[1]
    bases = lin_reprs.split(_sep)[1:]
    # now we try to find the form gA, B and tC
    ABC_forms = list()

    forms = base['forms']

    for format_form, base_rp, replace_key in zip(ABC, bases, key_words):
        found_root_form = None
        for root_form_lin_repr in _global_root_forms_lin_dict:
            check_form = _global_root_forms_lin_dict[root_form_lin_repr]
            check_temp = base_rp.replace(replace_key, check_form._pure_lin_repr)
            if check_temp == format_form:
                found_root_form = check_form
                break
            else:
                pass
        assert found_root_form is not None, f"must have found root-for for {format_form}."

        base_form = None
        for _pure_lin_repr in forms:
            if _pure_lin_repr == found_root_form._pure_lin_repr:
                base_form = forms[_pure_lin_repr]
                break
            else:
                pass
        assert base_form is not None, f"we must have found a msepy copy of the root-form."
        ABC_forms.append(base_form)

    return ABC_forms


def _find_space_through_pure_lin_repr(_target_space_lin_repr):
    """"""
    spaces = base['spaces']
    the_space = None
    for space_lin_repr in spaces:
        msepy_space = spaces[space_lin_repr]
        abs_space_pure_lin_repr = msepy_space.abstract._pure_lin_repr
        if abs_space_pure_lin_repr == _target_space_lin_repr:
            the_space = msepy_space
            break
        else:
            pass
    assert the_space is not None, f"Find no msepy space."
    return the_space


def _parse_root_form(root_form_vec_lin_repr):
    """"""
    forms = base['forms']
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
    return (
        dynamic_cochain_vec,
        rf.abstract.ap()._sym_repr,
        rf.cochain._ati_time_caller,
        rf.cochain._generation_caller,
    )


from msehy.tools.matrix.dynamic import IrregularDynamicLocalMatrix


def Parse__M_matrix(space, degree0, degree1):
    """"""
    degree0 = _str_degree_parser(degree0)
    degree1 = _str_degree_parser(degree1)

    the_space = _find_space_through_pure_lin_repr(space)

    if degree0 == degree1:
        M = the_space.mass_matrix.dynamic(degree0)
        assert M.__class__ is IrregularDynamicLocalMatrix, \
            f"mass matrix must be a irregular dynamic local matrix."
        return M, None, M._callable_data._generation_caller
        # time_indicator is None, mean M is same at all time.

    else:
        raise NotImplementedError()


def Parse__E_matrix(space, degree):
    """"""
    degree = _str_degree_parser(degree)
    the_space = _find_space_through_pure_lin_repr(space)
    E = the_space.incidence_matrix.dynamic(degree)
    assert E.__class__ is IrregularDynamicLocalMatrix, \
        f"mass matrix must be a irregular dynamic local matrix."
    return E, None, E._callable_data._generation_caller
    # time_indicator is None, mean M is same at all time.


    # gm0 = the_msepy_space.gathering_matrix._next(degree)
    # gm1 = the_msepy_space.gathering_matrix(degree)
    #
    # E = MsePyStaticLocalMatrix(  # make a new copy every single time.
    #     the_msepy_space.incidence_matrix(degree),
    #     gm0,
    #     gm1,
    # )
    #
    # return E, None  # time_indicator is None, mean E is same at all time.


# - (w x u, v) ------------------------------------------------------------------------------------
def Parse__astA_x_astB_ip_tC(gA, gB, tC):
    """(A X B, C), A and B are given, C is the test form, so it gives a dynamic vector."""

    ABC_forms = _find_from_bracket_ABC(_VarSetting_astA_x_astB_ip_tC, gA, gB, tC)
    _, _, C = ABC_forms
    nonlinear_operation = _AxBipC(*ABC_forms)
    c, time_caller, generation_caller = nonlinear_operation(1, C)
    return c, time_caller, generation_caller


def Parse__astA_x_B_ip_tC(gA, B, tC):
    """Remember, for this term, gA, b, tC must be root-forms."""

    ABC_forms = _find_from_bracket_ABC(_VarSetting_astA_x_B_ip_tC, gA, B, tC)
    A, B, C = ABC_forms  # A is given
    nonlinear_operation = _AxBipC(*ABC_forms)
    C = nonlinear_operation(2, C, B)

    return C, A.cochain._ati_time_caller, A.cochain._generation_caller
    # since A is given, its ati and generation determine the time and generation of C.


def Parse__A_x_astB_ip_tC(A, gB, tC):
    """"""
    ABC_forms = _find_from_bracket_ABC(_VarSetting_A_x_astB_ip_tC, A, gB, tC)
    A, B, C = ABC_forms  # B is given
    nonlinear_operation = _AxBipC(*ABC_forms)
    C = nonlinear_operation(2, C, A)
    return C, B.cochain._ati_time_caller, B.cochain._generation_caller
    # since B is given, its ati determine the time of C.
