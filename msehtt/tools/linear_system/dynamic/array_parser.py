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


root_array_lin_repr = _global_lin_repr_setting['array']
_front, _back = root_array_lin_repr
_len_front = len(_front)
_len_back = len(_back)
_len_transpose_text = len(_transpose_text)


def msehtt_root_array_parser(dls, array_lin_repr):
    """"""
    PARSER = dls._base['PARSER']

    if array_lin_repr[-_len_transpose_text:] == _transpose_text:
        transpose = True
        array_lin_repr = array_lin_repr[:-_len_transpose_text]
    else:
        transpose = False

    assert array_lin_repr[:_len_front] == _front and array_lin_repr[-_len_back:] == _back, \
        f"array_lin_repr={array_lin_repr} is not representing a root-array."
    array_lin_repr = array_lin_repr[_len_front:-_len_back]

    if array_lin_repr[-_len_rf_ap_lin_repr:] == _root_form_ap_lin_repr:
        # we are parsing a coefficient vector of a root-form
        assert transpose is False, 'should be this case.'
        # we are parsing a vector representing a root form.
        root_form_vec_lin_repr = array_lin_repr[:-_len_rf_ap_lin_repr]
        x, text, time_indicator = PARSER._parse_root_form(root_form_vec_lin_repr)
        return x, text, time_indicator

    else:

        indicators = array_lin_repr.split(_sep)  # these section represents all info of this root-array.
        type_indicator = indicators[0]           # this first one indicates the type
        info_indicators = indicators[1:]         # the _auxiliaries indicate the details.

        # ------------ basic ----------------------------------------------------------------------
        if type_indicator == PARSER._find_indicator(
                _VarSetting_mass_matrix):
            A, _ti = PARSER.Parse__M_matrix(*info_indicators)

        elif type_indicator == PARSER._find_indicator(
                _VarSetting_d_matrix):
            A, _ti = PARSER.Parse__E_matrix(*info_indicators)

        elif type_indicator == PARSER._find_indicator(
                _VarSetting_d_matrix_transpose):
            A, _ti = PARSER.Parse__E_matrix(*info_indicators)
            A = A.T

        elif type_indicator == PARSER._find_indicator(
                _VarSetting_boundary_dp_vector):
            A, _ti = PARSER.Parse__trStar_rf0_dp_tr_s1_vector(dls, *info_indicators)

        # ------------ (A x B, C) ----------------------------------------------------------------
        elif type_indicator == PARSER._find_indicator(
                       _VarSetting_astA_x_astB_ip_tC):
            A, _ti = PARSER.Parse__astA_x_astB_ip_tC(*info_indicators)

        elif type_indicator == PARSER._find_indicator(
                       _VarSetting_astA_x_B_ip_tC):
            A, _ti = PARSER.Parse__astA_x_B_ip_tC(*info_indicators)

        elif type_indicator == PARSER._find_indicator(
                       _VarSetting_A_x_astB_ip_tC):
            A, _ti = PARSER.Parse__A_x_astB_ip_tC(*info_indicators)

        # ------------ (A x B | C) ----------------------------------------------------------------
        elif type_indicator == PARSER._find_indicator(
                       _VarSetting_astA_x_astB__dp__tC):
            A, _ti = PARSER.Parse__astA_x_astB__dp__tC(*info_indicators)

        elif type_indicator == PARSER._find_indicator(
                       _VarSetting_astA_x_B__dp__tC):
            A, _ti = PARSER.Parse__astA_x_B__dp__tC(*info_indicators)

        elif type_indicator == PARSER._find_indicator(
                       _VarSetting_A_x_astB__dp__tC):
            A, _ti = PARSER.Parse__A_x_astB__dp__tC(*info_indicators)

        # ===========================================================================================
        else:
            raise NotImplementedError(
                f"I cannot parse array: {type_indicator} with parameters: {info_indicators}."
            )

        text = PARSER._indicator_templates[type_indicator]['symbol']

        if transpose:
            return A.T, text + r"^{\mathsf{T}}", _ti
        else:
            return A, text, _ti
