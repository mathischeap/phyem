# -*- coding: utf-8 -*-
r"""
"""

_setting_ = {
    'base': None
}

from phyem.src.spaces.main import _VarSetting_A_x_B_ip_C
from phyem.src.spaces.main import _VarSetting_A_x_B__dp__C
from phyem.src.spaces.main import _VarSetting_AxB_ip_dC
from phyem.src.spaces.main import _VarSetting_AB_ip_dC
from phyem.src.spaces.main import _VarSetting_AB_dp_dC
from phyem.src.spaces.main import _VarSetting_AB_dp_C
from phyem.src.spaces.main import _VarSetting_AB_ip_C
from phyem.src.spaces.main import _VarSetting_A_ip_BC

from phyem.msehtt.static.implementation_array_parser import _find_from_bracket_ABC

from phyem.msehtt.static.form.addons.nop_data_computer.trilinear_AxB_ip_C import AxB_ip_C
from phyem.msehtt.static.form.addons.nop_data_computer.trilinear_AxB_ip_dC import AxB_ip_dC
from phyem.msehtt.static.form.addons.nop_data_computer.trilinear_AB_ip_dC import AB_ip_dC
from phyem.msehtt.static.form.addons.nop_data_computer.trilinear_ABC import T_ABC


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


# -------- (AB, d(C)) ------------------------------------------------------------------
def __AB_ip_dC__(A, B, C):
    r""""""
    A, B, C = _find_from_bracket_ABC(_VarSetting_AB_ip_dC, A, B, C)
    noc = AB_ip_dC(A, B, C)
    X = noc(3)
    text = (
        rf"\left({A.abstract._sym_repr} {B.abstract._sym_repr}," +
        r"\mathrm{d}" + rf"{C.abstract._sym_repr}\right)"
    )
    return X, X._time_caller, text


# -------- <AB|d(C)> ------------------------------------------------------------------
def __AB_dp_dC__(A, B, C):
    r""""""
    A, B, C = _find_from_bracket_ABC(_VarSetting_AB_dp_dC, A, B, C)
    noc = AB_ip_dC(A, B, C)   # not a typo, use AB_ip_dC is correct.
    X = noc(3)
    text = (
        rf"\left<\left.{A.abstract._sym_repr} {B.abstract._sym_repr}\right|" +
        r"\mathrm{d}" + rf"{C.abstract._sym_repr}\right>"
    )
    return X, X._time_caller, text


# -------- (A, BC) ------------------------------------------------------------------
def __A_ip_BC__(A, B, C):
    r""""""
    A, B, C = _find_from_bracket_ABC(_VarSetting_A_ip_BC, A, B, C)
    noc = T_ABC(A, B, C)
    X = noc(3)
    text = (
        rf"\left({A.abstract._sym_repr}, {B.abstract._sym_repr} {C.abstract._sym_repr}\right)"
    )
    return X, X._time_caller, text


# -------- (AB, C) ------------------------------------------------------------------
def __AB_ip_C__(A, B, C):
    r""""""
    A, B, C = _find_from_bracket_ABC(_VarSetting_AB_ip_C, A, B, C)
    noc = T_ABC(A, B, C)
    X = noc(3)
    text = (
        rf"\left({A.abstract._sym_repr} {B.abstract._sym_repr}, {C.abstract._sym_repr}\right)"
    )
    return X, X._time_caller, text


# -------- <AB|C> ------------------------------------------------------------------
def __AB_dp_C__(A, B, C):
    r""""""
    A, B, C = _find_from_bracket_ABC(_VarSetting_AB_dp_C, A, B, C)
    noc = T_ABC(A, B, C)
    X = noc(3)
    text = (
        rf"\left<\left.{A.abstract._sym_repr} {B.abstract._sym_repr}\right| {C.abstract._sym_repr}\right>"
    )
    return X, X._time_caller, text

# ====================================================================================
