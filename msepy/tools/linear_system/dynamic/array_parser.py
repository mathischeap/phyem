# -*- coding: utf-8 -*-
r"""
"""
from src.spaces.main import *
from src.spaces.main import _sep
from src.config import _global_lin_repr_setting
from src.config import _transpose_text

from src.config import _root_form_ap_vec_setting

_root_form_ap_lin_repr = _root_form_ap_vec_setting['lin']
_len_rf_ap_lin_repr = len(_root_form_ap_lin_repr)

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
        type_indicator = indicators[0]           # this first one indicates the type
        info_indicators = indicators[1:]         # the _auxiliaries indicate the details.

        # Mass matrices, incidence matrices ---------------------------------
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

        elif type_indicator == _find_indicator(
                _VarSetting_pi_matrix):
            A, _ti = Parse__pi_matrix(*info_indicators)

        # natural bc vector =================================================
        elif type_indicator == _find_indicator(
                _VarSetting_boundary_dp_vector):
            A, _ti = Parse__trStar_rf0_dp_tr_s1_vector(dls, *info_indicators)

        # (w x u, v) ========================================================
        elif type_indicator == _find_indicator(
                _VarSetting_astA_x_astB_ip_tC):
            A, _ti = Parse__astA_x_astB_ip_tC(*info_indicators)

        elif type_indicator == _find_indicator(
                _VarSetting_astA_x_B_ip_tC):
            A, _ti = Parse__astA_x_B_ip_tC(*info_indicators)

        elif type_indicator == _find_indicator(
                _VarSetting_A_x_astB_ip_tC):
            A, _ti = Parse__A_x_astB_ip_tC(*info_indicators)

        # (dA, B otimes C) ===================================================
        elif type_indicator == _find_indicator(
                _VarSetting_dastA_astA_tp_tC):         # !!!
            A, _ti = Parse__dastA_astA_tp_tC(*info_indicators)
        elif type_indicator == _find_indicator(
                _VarSetting_dastA_tB_tp_astA):
            A, _ti = Parse__dastA_tB_tp_astA(*info_indicators)
        elif type_indicator == _find_indicator(
                _VarSetting_dtA_astB_tp_astB):
            A, _ti = Parse__dtA_astB_tp_astB(*info_indicators)
        elif type_indicator == _find_indicator(
                _VarSetting_dA_B_tp_C__1Known):
            A, _ti = Parse__dA_B_tp_C__1Known(*info_indicators)
        elif type_indicator == _find_indicator(
                _VarSetting_dA_B_tp_C__2Known):
            A, _ti = Parse__dA_B_tp_C__2Known(*info_indicators)

        # (A, B otimes C) ====================================================
        elif type_indicator == _find_indicator(
                _VarSetting_A_B_tp_C__1Known):
            A, _ti = Parse__A_B_tp_C__1Known(*info_indicators)
        elif type_indicator == _find_indicator(
                _VarSetting_A_B_tp_C__2Known):
            A, _ti = Parse__A_B_tp_C__2Known(*info_indicators)

        # (bundle form, special diagonal bundle form) ========================
        elif type_indicator == _find_indicator(
                _VarSetting_IP_matrix_bf_db):
            A, _ti = Parse__IP_matrix_bf_db(*info_indicators)

        elif type_indicator == _find_indicator(
                _VarSetting_IP_matrix_db_bf):
            sp_db, sp_bf, d_db, d_bf = info_indicators
            M, _ti = Parse__IP_matrix_bf_db(sp_bf, sp_db, d_bf, d_db)
            A = M.T

        # --------------------------------------------------------------------
        else:
            raise NotImplementedError(
                f"I cannot parse: {array_lin_repr} of type {type_indicator}."
            )

        text = _indicator_templates[type_indicator]['symbol']

        if transpose:
            return A.T, text + r"^{\mathsf{T}}", _ti
        else:
            return A, text, _ti
