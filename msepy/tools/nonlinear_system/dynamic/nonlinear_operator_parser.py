# -*- coding: utf-8 -*-
r"""
"""
from src.config import _nonlinear_ap_test_form_repr

from src.wf.term.ap import TermNonLinearOperatorAlgebraicProxy

from msepy.tools.linear_system.dynamic._arr_par import _find_from_bracket_ABC

from msepy.main import base
from msepy.form.addons.noc.diff_3_entries_AxB_ip_C import _AxBipC
from msepy.form.addons.noc.diff_3_entries_dA_ip_BtpC import __dA_ip_BtpC__
from msepy.form.addons.noc.diff_3_entries_A__ip__B_tp_C import _A__ip__B_tp_C_


from src.spaces.main import _sep
from src.spaces.main import _VarSetting_A_x_B_ip_C
from src.spaces.main import _VarSetting_dA_B_tp_C
from src.spaces.main import _VarSetting_A_B_tp_C


def msepy_nonlinear_operator_parser(mda):
    """"""
    if mda.__class__ is TermNonLinearOperatorAlgebraicProxy:
        pure_lin_repr = mda._pure_lin_repr
        assert _nonlinear_ap_test_form_repr['lin'] in pure_lin_repr, f"The nonlinear term must be tested with tf."

        msepy_correspondence = list()  # we found the msepy corresponding forms.
        for rf in mda._correspondence:
            for msepy_form_repr in base['forms']:
                msepy_form = base['forms'][msepy_form_repr]
                if msepy_form.abstract is rf:
                    msepy_correspondence.append(msepy_form)
                else:
                    pass

        mda_pure_lin_repr = pure_lin_repr.split(_nonlinear_ap_test_form_repr['lin'])[0]

        indicators = mda_pure_lin_repr.split(_sep)  # these section represents all info of this root-array.
        type_indicator = indicators[0]   # this first one indicates the type
        info_indicators = indicators[1:]  # the _auxiliaries indicate the details.
        if type_indicator == _VarSetting_A_x_B_ip_C[1].split(_sep)[0]:
            M, time_indicator = _parse_AxB_C(*info_indicators)
            text = r'\mathsf{X}'
        elif type_indicator == _VarSetting_dA_B_tp_C[1].split(_sep)[0]:
            M, time_indicator = _parse_dA_B_tp_C(*info_indicators)
            text = r'\mathsf{dtp}'
        elif type_indicator == _VarSetting_A_B_tp_C[1].split(_sep)[0]:
            M, time_indicator = _parse_A_B_tp_C(*info_indicators)
            text = r'\mathsf{tp}'
        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError()

    return M, text, time_indicator


def _parse_AxB_C(A, B, C):
    """"""
    # lin_reprs = _VarSetting_A_x_B_ip_C[1]
    # base_Ar, base_Br, base_Cr = lin_reprs.split(_sep)[1:]
    # replace_keys = (r"{A}", r"{B}", r"{C}")
    # ABC_forms = list()
    # msepy_forms = base['forms']
    # for format_form, base_rp, replace_key in zip((A, B, C), (base_Ar, base_Br, base_Cr), replace_keys):
    #     found_root_form = None
    #     for root_form_lin_repr in _global_root_forms_lin_dict:
    #         check_form = _global_root_forms_lin_dict[root_form_lin_repr]
    #         check_temp = base_rp.replace(replace_key, check_form._pure_lin_repr)
    #         if check_temp == format_form:
    #             found_root_form = check_form
    #             break
    #         else:
    #             pass
    #     assert found_root_form is not None, f"must have found root-for for {format_form}."
    #
    #     msepy_base_form = None
    #     for _pure_lin_repr in msepy_forms:
    #         if _pure_lin_repr == found_root_form._pure_lin_repr:
    #             msepy_base_form = msepy_forms[_pure_lin_repr]
    #             break
    #         else:
    #             pass
    #     assert msepy_base_form is not None, f"we must have found a msepy copy of the root-form."
    #     ABC_forms.append(msepy_base_form)

    ABC_forms = _find_from_bracket_ABC(_VarSetting_A_x_B_ip_C, A, B, C)
    nonlinear_operation = _AxBipC(*ABC_forms)
    X = nonlinear_operation(3)

    return X, X._time_caller


def _parse_dA_B_tp_C(A, B, C):
    """"""
    ABC_forms = _find_from_bracket_ABC(_VarSetting_dA_B_tp_C, A, B, C)
    nonlinear_operation = __dA_ip_BtpC__(*ABC_forms)
    X = nonlinear_operation(3)
    return X, X._time_caller


def _parse_A_B_tp_C(A, B, C):
    """"""
    ABC_forms = _find_from_bracket_ABC(_VarSetting_A_B_tp_C, A, B, C)
    nonlinear_operation = _A__ip__B_tp_C_(*ABC_forms)
    X = nonlinear_operation(3)
    return X, X._time_caller
