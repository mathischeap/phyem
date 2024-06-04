# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.spaces.main import _str_degree_parser
from src.spaces.main import _sep
from src.config import _form_evaluate_at_repr_setting
_rf_evaluate_at_lin_repr = _form_evaluate_at_repr_setting['lin']

from src.form.main import _global_root_forms_lin_dict
from src.wf.mp.linear_system_bc import _NaturalBoundaryCondition

from msehtt.tools.matrix.static.local import MseHttStaticLocalMatrix
from msehtt.tools.vector.dynamic import MseHttDynamicLocalVector
from msehtt.tools.vector.static.local import MseHttStaticLocalVector


# noinspection PyUnresolvedReferences
from src.spaces.main import *
_locals = locals()

_indicator_templates = {}

_setting_ = {
    'base': dict()
}


def _indicator_check():
    for i, key in enumerate(_locals):
        if key[:12] == '_VarSetting_':
            default_setting = _locals[key]
            splits = default_setting[1].split(_sep)
            indicator = splits[0]
            key_words = splits[1:]
            symbol = default_setting[0]
            assert indicator not in _indicator_templates, 'repeated indicator found.'
            _indicator_templates[indicator] = {
                'key_words': key_words,
                'symbol': symbol,
            }


_indicator_check()

_indicator_cache = {}


def _find_indicator(default_setting):
    key = str(default_setting)
    if key in _indicator_cache:
        pass
    else:
        _indicator_cache[key] = default_setting[1].split(_sep)[0]
    return _indicator_cache[key]


def _base_spaces():
    """"""
    return _setting_['base']['spaces']


def _base_forms():
    """"""
    return _setting_['base']['forms']


def _base_meshes():
    """"""
    return _setting_['base']['meshes']


def _find_from_bracket_ABC(default_repr, *ABC, key_words=("{A}", "{B}", "{C}")):
    """"""
    lin_reprs = default_repr[1]
    bases = lin_reprs.split(_sep)[1:]
    # now we try to find the form A, B and C
    ABC_forms = list()

    all_forms = _base_forms()

    for format_form, base_rp, replace_key in zip(ABC, bases, key_words):
        found_root_form = None
        for root_form_lin_repr in _global_root_forms_lin_dict:
            check_form = _global_root_forms_lin_dict[root_form_lin_repr]
            check_temp = base_rp.replace(replace_key, check_form._pure_lin_repr)
            if check_temp == format_form:
                found_root_form = check_form
                break
            else:
                pass

        assert found_root_form is not None, f"must have found root-for for {format_form}."

        msepy_base_form = None
        for _pure_lin_repr in all_forms:
            if _pure_lin_repr == found_root_form._pure_lin_repr:
                msepy_base_form = all_forms[_pure_lin_repr]
                break
            else:
                pass

        assert msepy_base_form is not None, f"we must have found a msepy copy of the root-form."
        ABC_forms.append(msepy_base_form)

    return ABC_forms


def _find_space_through_pure_lin_repr(_target_space_lin_repr):
    """"""
    spaces = _base_spaces()
    the_msepy_space = None
    for space_lin_repr in spaces:
        msepy_space = spaces[space_lin_repr]
        abs_space_pure_lin_repr = msepy_space.abstract._pure_lin_repr
        if abs_space_pure_lin_repr == _target_space_lin_repr:
            the_msepy_space = msepy_space
            break
        else:
            pass
    assert the_msepy_space is not None, f"Find no msehtt static space."
    return the_msepy_space


def _parse_root_form(root_form_vec_lin_repr):
    """"""
    forms = _base_forms()
    rf = None   # the root-form

    for rf_pure_lin_repr in forms:
        if rf_pure_lin_repr == root_form_vec_lin_repr:
            rf = forms[rf_pure_lin_repr]
        else:
            pass

    assert rf is not None, f"DO NOT find a msepy root-form, something is wrong."

    if _rf_evaluate_at_lin_repr in rf.abstract._pure_lin_repr:
        assert rf._pAti_form['base_form'] is not None, f"must be a particular root-form!"
    else:  # it is a general (not for a specific time step for example) vector of the root-form.
        assert rf._pAti_form['base_form'] is None, f"must be a general root-form!"

    dynamic_cochain_vec = rf.cochain.dynamic_vec
    return dynamic_cochain_vec, rf.abstract.ap()._sym_repr, rf.cochain._ati_time_caller


_cache_M_ = {}


def Parse__M_matrix(space, degree0, degree1):
    """"""
    degree0 = _str_degree_parser(degree0)
    degree1 = _str_degree_parser(degree1)
    space = _find_space_through_pure_lin_repr(space)
    key = (space, degree0, degree1)
    if key in _cache_M_:
        return _cache_M_[key]
    else:
        if degree0 == degree1:
            degree = degree0
            gm = space.gathering_matrix(degree)
            m, cache_key_dict = space.mass_matrix(degree)
            M = MseHttStaticLocalMatrix(  # make a new copy every single time.
                m,
                gm,
                gm,
                cache_key=cache_key_dict
            )
            RETURN = M, None  # time_indicator is None, mean M is same at all time.
        else:
            raise NotImplementedError()

        _cache_M_[key] = RETURN
        return RETURN


def Parse__E_matrix(space, degree):
    """"""
    degree = _str_degree_parser(degree)
    msehtt_space = _find_space_through_pure_lin_repr(space)
    gm0 = msehtt_space.gathering_matrix._next(degree)
    gm1 = msehtt_space.gathering_matrix(degree)
    e, cache_key_dict = msehtt_space.incidence_matrix(degree)
    E = MseHttStaticLocalMatrix(  # make a new copy every single time.
        e,
        gm0,
        gm1,
        cache_key=cache_key_dict
    )
    return E, None  # time_indicator is None, mean E is same at all time.


def Parse__trStar_rf0_dp_tr_s1_vector(dls, tr_star_rf0, tr_rf1):
    """"""
    found_root_form = None
    temp = _VarSetting_boundary_dp_vector[1].split(_sep)[2]
    for root_form_lin_repr in _global_root_forms_lin_dict:
        check_form = _global_root_forms_lin_dict[root_form_lin_repr]
        check_temp = temp.replace('{f1}', check_form._pure_lin_repr)
        if check_temp == tr_rf1:
            found_root_form = check_form
            break
        else:
            pass
    assert found_root_form is not None, f"we must have found a root form!"

    rf1 = found_root_form
    rf1 = _base_forms()[rf1._pure_lin_repr]

    b_vector_caller = _TrStarRf0DualPairingTrS1(dls, tr_star_rf0, rf1)

    return (
        MseHttDynamicLocalVector(b_vector_caller),
        b_vector_caller._time_caller,  # _TrStarRf0DualPairingTrS1 has a time caller
        # which determines the time of the matrix
    )


class _TrStarRf0DualPairingTrS1(Frozen):
    """"""
    def __init__(self, dls, tr_star_rf0, rf1):
        """ < tr star fr0 | trace rf1 >"""
        self._tr_star_rf0 = tr_star_rf0
        self._rf1 = rf1
        temp = _VarSetting_boundary_dp_vector[1].split(_sep)[1]

        found_root_form = None

        for root_form_lin_repr in _global_root_forms_lin_dict:
            check_form = _global_root_forms_lin_dict[root_form_lin_repr]
            check_temp = temp.replace('{f0}', check_form._pure_lin_repr)

            if check_temp == tr_star_rf0:
                found_root_form = check_form
                break
            else:
                pass

        assert found_root_form is not None, f"we must have found a root form!"

        assert dls.bc is not None, \
            f"must provided something in 'dls.bc' such that this b vector can be determined!"

        if _rf_evaluate_at_lin_repr in found_root_form._pure_lin_repr:

            base_form = found_root_form._pAti_form['base_form']
            self._ati = found_root_form._pAti_form['ati']
            assert all([_ is not None for _ in [base_form, self._ati]]), \
                f"we must have found a base form and its abstract time instant."

        else:
            base_form = found_root_form
            self._ati = None

        found_natural_bc = None
        found_boundary_section_sym_repr = None
        for boundary_section in dls.bc:
            bcs = dls.bc[boundary_section]
            for bc in bcs:
                if bc.__class__ is _NaturalBoundaryCondition:

                    raw_natural_bc = bc

                    provided_root_form = raw_natural_bc._provided_root_form

                    if self._ati is None:  # provided_root_form is the base (general) form
                        if base_form._pure_lin_repr == provided_root_form._pure_lin_repr:
                            found_natural_bc = bc
                            found_boundary_section_sym_repr = boundary_section
                            break
                        else:
                            pass
                    else:  # provided_root_form is an abstract form
                        if found_root_form._pure_lin_repr == provided_root_form._pure_lin_repr:
                            found_natural_bc = bc
                            found_boundary_section_sym_repr = boundary_section
                            break
                        else:
                            pass
                else:
                    pass

        assert found_natural_bc is not None, f"we must have found a dls natural bc!"
        assert found_boundary_section_sym_repr is not None, f"We must have found an abstract manifold sym repr."

        self._dls_natural_bc = found_natural_bc

        self._partial_mesh_boundary_section = None
        partial_meshes = _base_meshes()
        for partial_mesh_sym_repr in partial_meshes:
            msehtt_partial_mesh = partial_meshes[partial_mesh_sym_repr]
            if msehtt_partial_mesh.abstract.manifold._sym_repr == found_boundary_section_sym_repr:
                self._partial_mesh_boundary_section = msehtt_partial_mesh
                break
        assert self._partial_mesh_boundary_section is not None, f"we must have found a msehtt partial boundary section"

        rf0 = None
        msehtt_forms = _base_forms()
        for _pure_lin_repr in msehtt_forms:
            if _pure_lin_repr == base_form._pure_lin_repr:
                rf0 = msehtt_forms[_pure_lin_repr]
                break
            else:
                pass
        assert rf0 is not None, f"we must have found a msepy root form."
        self._rf0 = rf0  # may be useful in the future! just store this information here!
        self._dls = dls
        self._freeze()

    def _time_caller(self, *args, **kwargs):
        if self._ati is None:
            # rf0 of <tr star rf0 | ~> must be a general (base) form
            t = args[0]
            assert isinstance(t, (int, float)), \
                f"for general root-form, I receive a real number!"

        else:
            # rf0 of <tr star rf0 | ~> must be an abstract (base) form, take ``t`` from its ati.
            t = self._ati(**kwargs)()

        return t

    def __call__(self, *args, **kwargs):
        """"""
        t = self._time_caller(*args, **kwargs)

        # find from all configurations the correct one for this natural BC ----------------------------------
        configurations = self._dls.config._configurations
        the_configuration = None
        for config in configurations[::-1]:  # search from rear.
            _type = config['type']
            if _type == 'natural bc':
                place = config['place']
                if place is self._partial_mesh_boundary_section:
                    config_root_form = config['root_form']
                    if self._rf0 is config_root_form:  # note that self._rf0 must be a base-form already
                        the_configuration = config
                        break
                    else:
                        pass
                else:
                    pass
            else:
                pass
        assert the_configuration is not None, f'BC: {self._dls_natural_bc} is not configured!'

        category = the_configuration['category']
        # ----- study the configuration to choose a correct way to use it --------------------------------------
        if category == 1:  # natural bc 1
            # in this case, we provide one function in args as the exact boundary bc function rf0,
            # not trace-star-rf0
            exact_function = the_configuration['condition']
        else:
            raise NotImplementedError()

        # ------ use the configuration to produce the correct vector data used for a static local vector ------
        if category == 1:
            data = self._rf1.bi.with_vc_over_boundary_section(
                t,
                exact_function,    # the vc
                self._partial_mesh_boundary_section  # over this msehtt partial mesh boundary section.
            )
        else:
            raise NotImplementedError(f"not implemented for natural bc of category = {category}.")

        # -- make the static local vector representing the natural bc at this particular time ``t`` ------------
        full_vector = MseHttStaticLocalVector(data, self._rf1.cochain.gathering_matrix)
        return full_vector


# - (A x B, C) ------------------------------------------------------------------------------------

from msehtt.static.form.addons.nop_data_computer.trilinear_AxB_ip_C import AxB_ip_C


def Parse__astA_x_astB_ip_tC(gA, gB, tC):
    """(*A X *B, C), A and B are given, C is the test form, so it gives a dynamic vector."""
    gA, gB, tC = _find_from_bracket_ABC(_VarSetting_astA_x_astB_ip_tC, gA, gB, tC)
    noc = AxB_ip_C(gA, gB, tC)
    v, time_caller = noc(1, tC)
    return v, time_caller


def Parse__astA_x_B_ip_tC(gA, B, tC):
    """Remember, for this term, gA, b, tC must be root-forms."""
    gA, B, tC = _find_from_bracket_ABC(_VarSetting_astA_x_B_ip_tC, gA, B, tC)
    noc = AxB_ip_C(gA, B, tC)
    M = noc(2, tC, B)
    return M, gA.cochain._ati_time_caller  # since A is given, its ati determine the time of C.


def Parse__A_x_astB_ip_tC(A, gB, tC):
    """"""
    A, gB, tC = _find_from_bracket_ABC(_VarSetting_A_x_astB_ip_tC, A, gB, tC)
    noc = AxB_ip_C(A, gB, tC)
    M = noc(2, tC, A)
    return M, gB.cochain._ati_time_caller  # since B is given, its ati determine the time of C.


# - (A x B | C) ------------------------------------------------------------------------------------

from msehtt.static.form.addons.nop_data_computer.trilinear_AxB_dp_C import AxB_dp_C


def Parse__astA_x_astB__dp__tC(gA, gB, tC):
    """"""
    gA, gB, tC = _find_from_bracket_ABC(_VarSetting_astA_x_astB__dp__tC, gA, gB, tC)
    noc = AxB_dp_C(gA, gB, tC)
    v, time_caller = noc(1, tC)
    return v, time_caller


def Parse__astA_x_B__dp__tC(gA, B, tC):
    """Remember, for this term, gA, b, tC must be root-forms."""
    gA, B, tC = _find_from_bracket_ABC(_VarSetting_astA_x_B__dp__tC, gA, B, tC)
    noc = AxB_dp_C(gA, B, tC)
    M = noc(2, tC, B)
    return M, gA.cochain._ati_time_caller  # since A is given, its ati determine the time of C.


def Parse__A_x_astB__dp__tC(A, gB, tC):
    """"""
    A, gB, tC = _find_from_bracket_ABC(_VarSetting_A_x_astB__dp__tC, A, gB, tC)
    noc = AxB_dp_C(A, gB, tC)
    M = noc(2, tC, A)
    return M, gB.cochain._ati_time_caller  # since B is given, its ati determine the time of C.
