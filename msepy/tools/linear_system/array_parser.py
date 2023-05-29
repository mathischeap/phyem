# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 5:21 PM on 5/10/2023
"""
from tools.frozen import Frozen
from src.spaces.main import _sep
from src.config import _global_lin_repr_setting
from src.spaces.main import _default_mass_matrix_reprs
from src.spaces.main import _default_d_matrix_reprs
from src.spaces.main import _default_d_matrix_transpose_reprs
from src.spaces.main import _default_boundary_dp_vector_repr

from src.spaces.main import _str_degree_parser

from src.config import _form_evaluate_at_repr_setting, _transpose_text

_rf_evaluate_at_lin_repr = _form_evaluate_at_repr_setting['lin']

from src.config import _root_form_ap_vec_setting

_root_form_ap_lin_repr = _root_form_ap_vec_setting['lin']
_len_rf_ap_lin_repr = len(_root_form_ap_lin_repr)

from msepy.main import base

from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix
from msepy.tools.vector.dynamic import MsePyDynamicLocalVector

root_array_lin_repr = _global_lin_repr_setting['array']
_front, _back = root_array_lin_repr
_len_front = len(_front)
_len_back = len(_back)
_len_transpose_text = len(_transpose_text)


def msepy_root_array_parser(array_lin_repr):
    """"""
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
        return _parse_root_form(root_form_vec_lin_repr)

    else:
        indicators = array_lin_repr.split(_sep)  # these section represents all info of this root-array.
        type_indicator = indicators[0]   # this first one indicates the type
        info_indicators = indicators[1:]  # the others indicate the details.

        if type_indicator == _default_mass_matrix_reprs[1].split(_sep)[0]:
            M = _parse_M_matrix(*info_indicators)
            text = r"\mathsf{M}"
        elif type_indicator == _default_d_matrix_reprs[1].split(_sep)[0]:
            M = _parse_E_matrix(*info_indicators)
            text = r"\mathsf{E}"
        elif type_indicator == _default_d_matrix_transpose_reprs[1].split(_sep)[0]:
            M = _parse_E_matrix(*info_indicators).T
            text = r"\mathsf{E}^{\mathsf{T}}"
        elif type_indicator == _default_boundary_dp_vector_repr[1].split(_sep)[0]:
            M = _parse_trStar_bf0_dp_tr_s1_vector(*info_indicators)
            text = r"\boldsymbol{b}"
        else:
            raise NotImplementedError(f"I cannot parse: {array_lin_repr} of type {type_indicator}")

        if transpose:
            return M.T, text + r"^{\mathsf{T}}"
        else:
            return M, text


def _parse_root_form(root_form_vec_lin_repr):
    """"""
    forms = base['forms']
    rf = None
    for rf_pure_lin_repr in forms:
        if rf_pure_lin_repr == root_form_vec_lin_repr:
            rf = forms[rf_pure_lin_repr]
        else:
            pass

    assert rf is not None, f"DO NOT find a msepy root-form, something is wrong."

    if _rf_evaluate_at_lin_repr in rf.abstract._pure_lin_repr:
        assert rf._pAti_form['base_form'] is not None, f"must be a particular root-form!"
        dynamic_cochain_vec = rf.cochain.dynamic_vec
        return dynamic_cochain_vec, rf.abstract.ap()._sym_repr

    else:  # it is a general (not for a specific time step for example) vector of the root-form.

        assert rf._pAti_form['base_form'] is None, f"must be a general root-form!"

        dynamic_cochain_vec = rf.cochain.dynamic_vec

        return dynamic_cochain_vec, rf.abstract.ap()._sym_repr


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


def _parse_trStar_bf0_dp_tr_s1_vector(tr_star_rf0, space1, degree):
    """"""
    degree = _str_degree_parser(degree)
    spaces = base['spaces']
    the_msepy_space1 = None
    for space_lin_repr in spaces:
        msepy_space = spaces[space_lin_repr]
        abs_space_pure_lin_repr = msepy_space.abstract._pure_lin_repr
        if abs_space_pure_lin_repr == space1:
            the_msepy_space1 = msepy_space
            break
        else:
            pass
    b_vector_caller = _TrStarBf0DualPairingTrS1(tr_star_rf0, the_msepy_space1, degree)
    return MsePyDynamicLocalVector(b_vector_caller)


class _TrStarBf0DualPairingTrS1(Frozen):
    """"""
    def __init__(self, tr_star_rf0, space1, degree):
        """"""
        self._tr_star_rf0 = tr_star_rf0
        self._space1 = space1
        self._degree = degree
        self._freeze()

    def __call__(self, *args, **kwargs):
        """"""
        raise NotImplementedError()
