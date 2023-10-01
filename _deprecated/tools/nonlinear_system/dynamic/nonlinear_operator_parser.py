# -*- coding: utf-8 -*-
r"""
"""
from src.config import _nonlinear_ap_test_form_repr

from src.wf.term.ap import TermNonLinearOperatorAlgebraicProxy

from msehy.tools.linear_system.dynamic._arr_par import _find_from_bracket_ABC

from src.spaces.main import _sep

from src.spaces.main import _VarSetting_A_x_B_ip_C
from msehy.py.operations.nonlinear.AxB_ip_C import _AxBipC


from msehy.py2.main import base as ___base___


def msehy_nonlinear_operator_parser(mda):
    """"""
    if mda.__class__ is TermNonLinearOperatorAlgebraicProxy:
        pure_lin_repr = mda._pure_lin_repr
        assert _nonlinear_ap_test_form_repr['lin'] in pure_lin_repr, f"The nonlinear term must be tested with tf."

        msepy_correspondence = list()  # we found the msepy corresponding forms.
        for rf in mda._correspondence:
            for msepy_form_repr in ___base___['forms']:
                msepy_form = ___base___['forms'][msepy_form_repr]
                if msepy_form.abstract is rf:
                    msepy_correspondence.append(msepy_form)
                else:
                    pass

        mda_pure_lin_repr = pure_lin_repr.split(_nonlinear_ap_test_form_repr['lin'])[0]

        indicators = mda_pure_lin_repr.split(_sep)  # these section represents all info of this root-array.
        type_indicator = indicators[0]   # this first one indicates the type
        info_indicators = indicators[1:]  # the others indicate the details.

        if (type_indicator
                == _VarSetting_A_x_B_ip_C[1].split(_sep)[0]):
            M, ti, gi = _parse_A_x_B_ip_C(*info_indicators)
            text = r'\mathsf{X}'
        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError()

    return M, text, ti, gi


def _parse_A_x_B_ip_C(A, B, C):
    """"""

    ABC_forms = _find_from_bracket_ABC(_VarSetting_A_x_B_ip_C, A, B, C)
    nonlinear_operation = _AxBipC(*ABC_forms)
    X = nonlinear_operation(3)

    return X, X._time_caller, X._generation_caller
