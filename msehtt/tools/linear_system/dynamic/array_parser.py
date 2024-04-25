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

import msehtt.tools.linear_system.dynamic._arr_par as PARSER

root_array_lin_repr = _global_lin_repr_setting['array']
_front, _back = root_array_lin_repr
_len_front = len(_front)
_len_back = len(_back)
_len_transpose_text = len(_transpose_text)

_local_config = {
    'indicator_check': False,
}


def msehtt_root_array_parser(dls, array_lin_repr):
    """"""
    if _local_config['indicator_check']:
        pass
    else:
        PARSER._indicator_check()
        _local_config['indicator_check'] = True

    if PARSER._setting_['base'] == {}:
        PARSER._setting_['base'] = dls._base
    else:
        assert PARSER._setting_['base'] is dls._base, f"parser base changed!"

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
        # return x, text, time_indicator
