# -*- coding: utf-8 -*-
r"""
"""

_setting_ = {
    'base': None
}

from src.spaces.main import _VarSetting_A_x_B_ip_C
from src.spaces.main import _VarSetting_A_x_B__dp__C
from src.spaces.main import _VarSetting_AxB_ip_dC

from msehtt.static.implementation_array_parser import _find_from_bracket_ABC

from msehtt.static.form.addons.nop_data_computer.trilinear_AxB_ip_C import AxB_ip_C
from msehtt.static.form.addons.nop_data_computer.trilinear_AxB_ip_dC import AxB_ip_dC


def __A_x_B_ip_C__(A, B, C):
    r"""(AxB, c)"""
    A, B, C = _find_from_bracket_ABC(_VarSetting_A_x_B_ip_C, A, B, C)
    noc = AxB_ip_C(A, B, C)
    X = noc(3)
    text = rf"\left({A.abstract._sym_repr}\times{B.abstract._sym_repr},{C.abstract._sym_repr}\right)"
    return X, X._time_caller, text


def __A_x_B_dp_C__(A, B, C):
    r"""<AxB|c>"""
    A, B, C = _find_from_bracket_ABC(_VarSetting_A_x_B__dp__C, A, B, C)
    noc = AxB_ip_C(A, B, C)
    X = noc(3)
    text = (rf"\left\langle\left."
            rf"{A.abstract._sym_repr}\times{B.abstract._sym_repr}\right|{C.abstract._sym_repr}\right\rangle")
    return X, X._time_caller, text


def __A_x_B_ip_dC__(A, B, C):
    r"""(AxB, dC)"""
    A, B, C = _find_from_bracket_ABC(_VarSetting_AxB_ip_dC, A, B, C)
    noc = AxB_ip_dC(A, B, C)
    X = noc(3)
    text = (
        rf"\left({A.abstract._sym_repr}\times{B.abstract._sym_repr}," +
        r"\mathrm{d}" + rf"{C.abstract._sym_repr}\right)"
    )
    return X, X._time_caller, text
