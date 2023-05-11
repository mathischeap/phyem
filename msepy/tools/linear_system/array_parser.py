# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 5:21 PM on 5/10/2023
"""
from src.spaces.main import _sep
from src.config import _global_lin_repr_setting
from src.spaces.main import _default_mass_matrix_reprs
from src.spaces.main import _default_d_matrix_reprs
from src.spaces.main import _default_d_matrix_transpose_reprs

from src.spaces.main import _str_degree_parser

from src.config import _form_evaluate_at_repr_setting
from src.config import _root_form_ap_vec_setting

_root_form_ap_lin_repr = _root_form_ap_vec_setting['lin']
_len_rf_ap_lin_repr = len(_root_form_ap_lin_repr)

from msepy.main import base

from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix

root_array_lin_repr = _global_lin_repr_setting['array']
_front, _back = root_array_lin_repr
_len_front = len(_front)
_len_back = len(_back)


def msepy_root_array_parser(array_lin_repr):
    """"""
    assert array_lin_repr[:_len_front] == _front and array_lin_repr[-_len_back:] == _back, \
        f"array_lin_repr={array_lin_repr} is not representing a root-array."
    array_lin_repr = array_lin_repr[_len_front:-_len_back]

    if array_lin_repr[-_len_rf_ap_lin_repr:] == _root_form_ap_lin_repr:
        # we are parsing a vector representing a root form.

        pass

    else:
        indicators = array_lin_repr.split(_sep)  # these section represents all info of this root-array.
        type_indicator = indicators[0]   # this first one indicates the type
        info_indicators = indicators[1:]  # the others indicate the details.

        if type_indicator == _default_mass_matrix_reprs[1].split(_sep)[0]:
            M = _parse_M_matrix(*info_indicators)
            return M, r"\mathsf{M}"
        elif type_indicator == _default_d_matrix_reprs[1].split(_sep)[0]:
            E = _parse_E_matrix(*info_indicators)
            return E, r"\mathsf{E}"
        elif type_indicator == _default_d_matrix_transpose_reprs[1].split(_sep)[0]:
            E = _parse_E_matrix(*info_indicators)
            return E.T, r"\mathsf{E}^{\mathsf{T}}"

        else:
            raise NotImplementedError(f"I cannot parse: {array_lin_repr}")


def _parse_M_matrix(space, degree0, degree1):
    """"""
    degree0 = _str_degree_parser(degree0)
    degree1 = _str_degree_parser(degree1)
    spaces = base['spaces']
    the_msepy_space = None
    for space_lin_repr in spaces:
        msepy_space = spaces[space_lin_repr]
        abs_space_pure_lin_repr = msepy_space.abstract._pure_lin_repr
        if abs_space_pure_lin_repr == space:
            the_msepy_space = msepy_space
            break
        else:
            pass
    if degree0 == degree1:
        degree = degree0
        gm = the_msepy_space.gathering_matrix(degree)
        M = MsePyStaticLocalMatrix(  # make a new copy every single time.
            the_msepy_space.mass_matrix(degree),
            gm,
            gm,
        )
        return M
    else:
        raise NotImplementedError()


def _parse_E_matrix(space, degree):
    """"""
    degree = _str_degree_parser(degree)
    spaces = base['spaces']
    the_msepy_space = None
    for space_lin_repr in spaces:
        msepy_space = spaces[space_lin_repr]
        abs_space_pure_lin_repr = msepy_space.abstract._pure_lin_repr
        if abs_space_pure_lin_repr == space:
            the_msepy_space = msepy_space
            break
        else:
            pass

    gm0 = the_msepy_space.gathering_matrix._next(degree)
    gm1 = the_msepy_space.gathering_matrix(degree)
    E = MsePyStaticLocalMatrix(  # make a new copy every single time.
        the_msepy_space.incidence_matrix(degree),
        gm0,
        gm1,
    )
    return E


if __name__ == '__main__':
    # python msepy/tools/linear_system/array_parser.py
    pass
