# -*- coding: utf-8 -*-
"""
"""

_setting_ = {
    'base': None
}

from src.spaces.main import _VarSetting_A_x_B_ip_C

from msehtt.static.implementation_array_parser import _find_from_bracket_ABC
from msehtt.static.form.addons.nop_data_computer.trilinear_AxB_ip_C import AxB_ip_C


def __A_x_B_ip_C__(A, B, C):
    """"""
    A, B, C = _find_from_bracket_ABC(_VarSetting_A_x_B_ip_C, A, B, C)
    noc = AxB_ip_C(A, B, C)
    X = noc(3)
    text = rf"\left\langle{A.abstract._sym_repr}\times{B.abstract._sym_repr},{C.abstract._sym_repr}\right\rangle"
    return X, X._time_caller, text
