# -*- coding: utf-8 -*-
r"""
"""
from src.spaces.main import *
from src.spaces.main import _sep
from src.config import _global_lin_repr_setting
from msehy.tools.linear_system.dynamic._arr_par import *


from src.config import _transpose_text
_len_transpose_text = len(_transpose_text)

root_array_lin_repr = _global_lin_repr_setting['array']
_front, _back = root_array_lin_repr
_len_front = len(_front)
_len_back = len(_back)

from src.config import _root_form_ap_vec_setting

_root_form_ap_lin_repr = _root_form_ap_vec_setting['lin']
_len_rf_ap_lin_repr = len(_root_form_ap_lin_repr)


_local_config = {
    'indicator_check': False
}


def msehy_root_array_parser(dls, array_lin_repr):
    """"""
    assert dls.__class__.__name__ == 'IrregularDynamicLinearSystem', \
        f'must call me for a IrregularDynamicLinearSystem!'

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
        x, text, time_indicator, generation_indicator = _parse_root_form(root_form_vec_lin_repr)
        return x, text, time_indicator, generation_indicator

    else:

        indicators = array_lin_repr.split(_sep)  # these section represents all info of this root-array.
        type_indicator = indicators[0]           # this first one indicates the type
        info_indicators = indicators[1:]         # the others indicate the details.

        # Mass matrices, incidence matrices ---------------------------------
        if type_indicator == _find_indicator(
                _VarSetting_mass_matrix):
            A, _ti, _gi = Parse__M_matrix(*info_indicators)

        elif type_indicator == _find_indicator(
                _VarSetting_d_matrix):
            A, _ti, _gi = Parse__E_matrix(*info_indicators)

        elif type_indicator == _find_indicator(
                _VarSetting_d_matrix_transpose):
            A, _ti, _gi = Parse__E_matrix(*info_indicators)
            A = A.T

        # (w x u, v) ========================================================
        elif type_indicator == _find_indicator(
                _VarSetting_astA_x_astB_ip_tC):
            A, _ti, _gi = Parse__astA_x_astB_ip_tC(*info_indicators)

        elif type_indicator == _find_indicator(
                _VarSetting_astA_x_B_ip_tC):
            A, _ti, _gi = Parse__astA_x_B_ip_tC(*info_indicators)

        elif type_indicator == _find_indicator(
                _VarSetting_A_x_astB_ip_tC):
            A, _ti, _gi = Parse__A_x_astB_ip_tC(*info_indicators)

        # --------------------------------------------------------------------
        else:
            raise NotImplementedError(
                f"I cannot parse: {array_lin_repr} of type {type_indicator}."
            )

        text = _indicator_templates[type_indicator]['symbol']

        if transpose:
            return A.T, text + r"^{\mathsf{T}}", _ti, _gi
        else:
            return A, text, _ti, _gi