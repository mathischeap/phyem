# -*- coding: utf-8 -*-
r"""
"""
from src.spaces.main import _sep
from src.config import _global_lin_repr_setting

from src.spaces.main import _str_degree_parser
from src.spaces.main import *

from src.config import _form_evaluate_at_repr_setting, _transpose_text

_rf_evaluate_at_lin_repr = _form_evaluate_at_repr_setting['lin']

from src.config import _root_form_ap_vec_setting

_root_form_ap_lin_repr = _root_form_ap_vec_setting['lin']
_len_rf_ap_lin_repr = len(_root_form_ap_lin_repr)

from msepy.form.tools.operations.nonlinear.dA_ip_BtpC import __dA_ip_BtpC__
from msepy.form.tools.operations.nonlinear.AtpB_C import ___AtpB_C__

from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix

from msepy.tools.linear_system.dynamic._arr_par import *

root_array_lin_repr = _global_lin_repr_setting['array']
_front, _back = root_array_lin_repr
_len_front = len(_front)
_len_back = len(_back)
_len_transpose_text = len(_transpose_text)

_local_config = {
    'indicator_check': False
}


def msepy_root_array_parser(dls, array_lin_repr):
    """"""
    if _local_config['indicator_check']:
        pass
    else:
        _indicator_check()
        _local_config['indicator_check'] = True

    if array_lin_repr[-_len_transpose_text:] == _transpose_text:
        transpose = True
        array_lin_repr = array_lin_repr[:-_len_transpose_text]
    else:
        transpose = False

    assert array_lin_repr[:_len_front] == _front and array_lin_repr[-_len_back:] == _back, \
        f"array_lin_repr={array_lin_repr} is not representing a root-array."
    array_lin_repr = array_lin_repr[_len_front:-_len_back]

    if array_lin_repr[-_len_rf_ap_lin_repr:] == _root_form_ap_lin_repr:
        assert transpose is False, 'should be this case.'
        # we are parsing a vector representing a root form.
        root_form_vec_lin_repr = array_lin_repr[:-_len_rf_ap_lin_repr]
        x, text, time_indicator = _parse_root_form(root_form_vec_lin_repr)
        return x, text, time_indicator

    else:

        indicators = array_lin_repr.split(_sep)  # these section represents all info of this root-array.
        type_indicator = indicators[0]   # this first one indicates the type
        info_indicators = indicators[1:]  # the others indicate the details.

        # Mass matrices, incidence matrices -----------------------------------------
        if type_indicator == _find_indicator(
                _VarSetting_mass_matrix):
            A, _ti = Parse__M_matrix(*info_indicators)

        elif type_indicator == _find_indicator(
                _VarSetting_d_matrix):
            A, _ti = Parse__E_matrix(*info_indicators)

        elif type_indicator == _find_indicator(
                _VarSetting_d_matrix_transpose):
            A, _ti = Parse__E_matrix(*info_indicators)
            A = A.T

        # natural bc vector ======================================================
        elif type_indicator == _find_indicator(
                _VarSetting_boundary_dp_vector):
            A, _ti = Parse__trStar_rf0_dp_tr_s1_vector(dls, *info_indicators)

        # (w x u, v) =============================================================
        elif type_indicator == _find_indicator(
                _VarSetting_astA_x_astB_ip_tC):
            A, _ti = Parse__astA_x_astB_ip_tC(*info_indicators)

        elif type_indicator == _find_indicator(
                _VarSetting_astA_x_B_ip_tC):
            A, _ti = Parse__astA_x_B_ip_tC(*info_indicators)

        elif type_indicator == _find_indicator(
                _VarSetting_A_x_astB_ip_tC):
            A, _ti = Parse__A_x_astB_ip_tC(*info_indicators)

        # (dA, B otimes C) ======================================================== !
        elif type_indicator == _find_indicator(
                _VarSetting_dastA_B_tp_tC):
            A, _ti = Parse__dastA_B_tp_tC(*info_indicators)

        elif type_indicator == _find_indicator(
                _VarSetting_dastA_astA_tp_tC):
            A, _ti = Parse__dastA_astA_tp_tC(*info_indicators)

        # (bundle form, special diagonal bundle form) ============================== !
        elif type_indicator == _find_indicator(
                _VarSetting_mass_matrix_bf_db):
            A, _ti = Parse__mass_matrix_bf_db(*info_indicators)

        elif type_indicator == _find_indicator(
                _VarSetting_mass_matrix_db_bf):
            sp_db, sp_bf, d_db, d_bf = info_indicators
            M, _ti = Parse__mass_matrix_bf_db(sp_bf, sp_db, d_bf, d_db)
            A = M.T

        # (A otimes B, C) =========================================================== !
        elif type_indicator == _find_indicator(
                _VarSetting_astA_tp_B_tC):
            A, _ti = Parse__astA_tp_B_tC(*info_indicators)

        elif type_indicator == _find_indicator(
                _VarSetting_astA_tp_astA_tC):
            A, _ti = Parse__astA_tp_astA_tC(*info_indicators)

        # ============================================================================
        else:
            raise NotImplementedError(f"I cannot parse: {array_lin_repr} of type {type_indicator}")

        text = _indicator_templates[type_indicator]['symbol']

        if transpose:
            return A.T, text + r"^{\mathsf{T}}", _ti
        else:
            return A, text, _ti


def Parse__mass_matrix_bf_db(sp_bf, sp_db, d_bf, d_db):
    """"""
    sp_bf = _find_space_through_pure_lin_repr(sp_bf)
    sp_db = _find_space_through_pure_lin_repr(sp_db)
    d_bf = _str_degree_parser(d_bf)
    d_db = _str_degree_parser(d_db)

    M = sp_bf[d_bf].inner_product(sp_db[d_db], special_key=0)

    gm_row = sp_bf.gathering_matrix(d_bf)
    gm_col = sp_db.gathering_matrix(d_db)

    M = MsePyStaticLocalMatrix(  # make a new copy every single time.
        M,
        gm_row,
        gm_col,
    )

    return M, None  # time_indicator is None, mean M is same at all time.


def Parse__dastA_B_tp_tC(gA, B, tC):
    """"""
    ABC_forms = _find_from_bracket_ABC(_VarSetting_dastA_B_tp_tC, gA, B, tC)
    msepy_A, msepy_B, msepy_C = ABC_forms  # A is given
    nonlinear_operation = __dA_ip_BtpC__(*ABC_forms)
    C = nonlinear_operation(2, msepy_C, msepy_B)
    return C, msepy_A.cochain._ati_time_caller  # since A is given, its ati determine the time of C.


def Parse__dastA_astA_tp_tC(gA, tC):
    """"""
    AC_forms = _find_from_bracket_ABC(_VarSetting_dastA_astA_tp_tC, gA, tC, key_words=("{A}", "{C}"))
    gA, tC = AC_forms
    nonlinear_operation = __dA_ip_BtpC__(gA, gA, tC)
    c, time_caller = nonlinear_operation(1, tC)
    return c, time_caller  # since A is given, its ati determine the time of tC.


def Parse__astA_tp_astA_tC(gA, tC):
    AC_forms = _find_from_bracket_ABC(_VarSetting_dastA_astA_tp_tC, gA, tC, key_words=("{A}", "{C}"))
    gA, tC = AC_forms
    nonlinear_operation = ___AtpB_C__(gA, gA, tC)
    c, time_caller = nonlinear_operation(1, tC)
    return c, time_caller  # since A is given, its ati determine the time of tC.


def Parse__astA_tp_B_tC(gA, B, tC):
    """"""
    ABC_forms = _find_from_bracket_ABC(_VarSetting_astA_tp_B_tC, gA, B, tC)
    msepy_A, msepy_B, msepy_C = ABC_forms  # A is given
    nonlinear_operation = ___AtpB_C__(*ABC_forms)
    C = nonlinear_operation(2, msepy_C, msepy_B)
    return C, msepy_A.cochain._ati_time_caller  # since A is given, its ati determine the time of C.
