# -*- coding: utf-8 -*-
"""
"""
from src.config import _nonlinear_ap_test_form_repr
from src.wf.term.ap import TermNonLinearOperatorAlgebraicProxy

from src.spaces.main import _sep
from src.spaces.main import _VarSetting_A_x_B_ip_C


def nonlinear_operator_parser(noc_term, imp_base):
    """"""
    PARSER = imp_base['NOC-PARSER']
    if noc_term.__class__ is TermNonLinearOperatorAlgebraicProxy:
        pure_lin_repr = noc_term._pure_lin_repr
        assert _nonlinear_ap_test_form_repr['lin'] in pure_lin_repr, f"The nonlinear term must be tested with tf."

        correspondence = list()  # we found the msepy corresponding forms.
        for rf in noc_term._correspondence:
            for form_repr in imp_base['forms']:
                form = imp_base['forms'][form_repr]
                if form.abstract is rf:
                    correspondence.append(form)
                else:
                    pass

        term_pure_lin_repr = pure_lin_repr.split(_nonlinear_ap_test_form_repr['lin'])[0]

        indicators = term_pure_lin_repr.split(_sep)  # these section represents all info of this root-array.
        type_indicator = indicators[0]   # this first one indicates the type
        info_indicators = indicators[1:]  # the _auxiliaries indicate the details.

        if type_indicator == _VarSetting_A_x_B_ip_C[1].split(_sep)[0]:
            M, time_indicator, text = PARSER.__A_x_B_ip_C__(*info_indicators)
        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError()

    return M, text, time_indicator
