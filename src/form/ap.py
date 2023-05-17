# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
created at: 3/20/2023 3:57 PM
"""
from src.config import _form_evaluate_at_repr_setting, _root_form_ap_vec_setting
from src.algebra.array import _root_array


def _parse_root_form_ap(f, sym_repr=None):
    """"""
    assert f.is_root(), f"safety check."
    if sym_repr is None:
        setting_sym = _root_form_ap_vec_setting['sym']
        if f._pAti_form['base_form'] is None:
            sym_repr = setting_sym[0] + f._sym_repr + setting_sym[1]
        else:
            ss = _form_evaluate_at_repr_setting['sym']
            sym_repr = f._sym_repr.split(ss[1])[0].split(ss[0])[1]
            sym_repr = setting_sym[0] + sym_repr + setting_sym[1]
            sym_repr = ss[0] + sym_repr + ss[1] + f._sym_repr.split(ss[1])[1]
    else:
        pass

    lr = f._pure_lin_repr + _root_form_ap_vec_setting['lin']

    return _root_array(sym_repr, lr, (f._ap_shape(), 1))
