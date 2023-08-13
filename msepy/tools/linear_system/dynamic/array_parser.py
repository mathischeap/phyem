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
from src.spaces.main import _default_boundary_dp_vector_reprs
from src.spaces.main import _default_astA_x_astB_ip_tC_reprs
from src.spaces.main import _default_astA_x_B_ip_tC_reprs
from src.spaces.main import _default_A_x_astB_ip_tC_reprs

from src.spaces.main import _str_degree_parser

from src.form.main import _global_root_forms_lin_dict

from src.config import _form_evaluate_at_repr_setting, _transpose_text

_rf_evaluate_at_lin_repr = _form_evaluate_at_repr_setting['lin']

from src.config import _root_form_ap_vec_setting

_root_form_ap_lin_repr = _root_form_ap_vec_setting['lin']
_len_rf_ap_lin_repr = len(_root_form_ap_lin_repr)

from msepy.main import base
from msepy.form.tools.operations.nonlinear.AxB_ip_C import _AxBipC

from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix
from msepy.tools.vector.dynamic import MsePyDynamicLocalVector
from msepy.tools.vector.static.local import MsePyStaticLocalVector

from msepy.tools.linear_system.dynamic.bc import MsePyDLSNaturalBoundaryCondition
from msepy.mesh.boundary_section.main import MsePyBoundarySectionMesh

root_array_lin_repr = _global_lin_repr_setting['array']
_front, _back = root_array_lin_repr
_len_front = len(_front)
_len_back = len(_back)
_len_transpose_text = len(_transpose_text)


def msepy_root_array_parser(dls, array_lin_repr):
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
        x, text, time_indicator = _parse_root_form(root_form_vec_lin_repr)
        return x, text, time_indicator

    else:

        indicators = array_lin_repr.split(_sep)  # these section represents all info of this root-array.
        type_indicator = indicators[0]   # this first one indicates the type
        info_indicators = indicators[1:]  # the others indicate the details.

        if type_indicator == _default_mass_matrix_reprs[1].split(_sep)[0]:
            A, time_indicator = _parse_M_matrix(*info_indicators)
            text = r"\mathsf{M}"
        elif type_indicator == _default_d_matrix_reprs[1].split(_sep)[0]:
            A, time_indicator = _parse_E_matrix(*info_indicators)
            text = r"\mathsf{E}"
        elif type_indicator == _default_d_matrix_transpose_reprs[1].split(_sep)[0]:
            A, time_indicator = _parse_E_matrix(*info_indicators)
            A = A.T
            text = r"\mathsf{E}^{\mathsf{T}}"
        elif type_indicator == _default_boundary_dp_vector_reprs[1].split(_sep)[0]:
            A, time_indicator = _parse_trStar_rf0_dp_tr_s1_vector(dls, *info_indicators)
            text = r"\boldsymbol{b}"
        elif type_indicator == _default_astA_x_astB_ip_tC_reprs[1].split(_sep)[0]:
            A, time_indicator = _parse_astA_x_astB_ip_tC(*info_indicators)
            text = r"\mathsf{c}"

        elif type_indicator == _default_astA_x_B_ip_tC_reprs[1].split(_sep)[0]:
            A, time_indicator = _parse_astA_x_B_ip_tC(*info_indicators)
            text = r"\mathsf{C}"
        elif type_indicator == _default_A_x_astB_ip_tC_reprs[1].split(_sep)[0]:
            A, time_indicator = _parse_A_x_astB_ip_tC(*info_indicators)
            text = r"\boldsymbol{C}"
        else:
            raise NotImplementedError(f"I cannot parse: {array_lin_repr} of type {type_indicator}")

        if transpose:
            return A.T, text + r"^{\mathsf{T}}", time_indicator
        else:
            return A, text, time_indicator


def _parse_root_form(root_form_vec_lin_repr):
    """"""
    msepy_forms = base['forms']
    rf = None   # msepy rf
    for rf_pure_lin_repr in msepy_forms:
        if rf_pure_lin_repr == root_form_vec_lin_repr:
            rf = msepy_forms[rf_pure_lin_repr]
        else:
            pass

    assert rf is not None, f"DO NOT find a msepy root-form, something is wrong."

    if _rf_evaluate_at_lin_repr in rf.abstract._pure_lin_repr:
        assert rf._pAti_form['base_form'] is not None, f"must be a particular root-form!"
        dynamic_cochain_vec = rf.cochain.dynamic_vec
        return dynamic_cochain_vec, rf.abstract.ap()._sym_repr, rf.cochain._ati_time_caller

    else:  # it is a general (not for a specific time step for example) vector of the root-form.
        assert rf._pAti_form['base_form'] is None, f"must be a general root-form!"
        dynamic_cochain_vec = rf.cochain.dynamic_vec
        return dynamic_cochain_vec, rf.abstract.ap()._sym_repr, rf.cochain._ati_time_caller


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

        return M, None  # time_indicator is None, mean M is same at all time.

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

    return E, None  # time_indicator is None, mean E is same at all time.


def _parse_astA_x_astB_ip_tC(gA, gB, tC):
    """(A X B, C), A and B are given, C is the test form, so it gives a dynamic vector."""
    lin_reprs = _default_astA_x_astB_ip_tC_reprs[1]
    base_Ar, base_Br, base_Cr = lin_reprs.split(_sep)[1:]
    replace_keys = (r"{A}", r"{B}", r"{C}")
    # now we try to find the form gA, B and tC
    ABC_forms = list()

    msepy_forms = base['forms']

    for format_form, base_rp, replace_key in zip((gA, gB, tC), (base_Ar, base_Br, base_Cr), replace_keys):
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
        for _pure_lin_repr in msepy_forms:
            if _pure_lin_repr == found_root_form._pure_lin_repr:
                msepy_base_form = msepy_forms[_pure_lin_repr]
                break
            else:
                pass
        assert msepy_base_form is not None, f"we must have found a msepy copy of the root-form."
        ABC_forms.append(msepy_base_form)

    _, _, msepy_C = ABC_forms  # A is given

    nonlinear_operation = _AxBipC(*ABC_forms)

    c, time_caller = nonlinear_operation(1, msepy_C)

    return c, time_caller  # since A is given, its ati determine the time of C.


def _parse_astA_x_B_ip_tC(gA, B, tC):
    """Remember, for this term, gA, b, tC must be root-forms."""
    lin_reprs = _default_astA_x_B_ip_tC_reprs[1]
    base_Ar, base_Br, base_Cr = lin_reprs.split(_sep)[1:]
    replace_keys = (r"{A}", r"{B}", r"{C}")
    # now we try to find the form gA, B and tC
    ABC_forms = list()

    msepy_forms = base['forms']

    for format_form, base_rp, replace_key in zip((gA, B, tC), (base_Ar, base_Br, base_Cr), replace_keys):
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
        for _pure_lin_repr in msepy_forms:
            if _pure_lin_repr == found_root_form._pure_lin_repr:
                msepy_base_form = msepy_forms[_pure_lin_repr]
                break
            else:
                pass
        assert msepy_base_form is not None, f"we must have found a msepy copy of the root-form."
        ABC_forms.append(msepy_base_form)

    msepy_A, msepy_B, msepy_C = ABC_forms  # A is given
    nonlinear_operation = _AxBipC(*ABC_forms)

    C = nonlinear_operation(2, msepy_C, msepy_B)

    return C, msepy_A.cochain._ati_time_caller  # since A is given, its ati determine the time of C.


def _parse_A_x_astB_ip_tC(A, gB, tC):
    """"""
    lin_reprs = _default_A_x_astB_ip_tC_reprs[1]
    base_Ar, base_Br, base_Cr = lin_reprs.split(_sep)[1:]
    replace_keys = (r"{A}", r"{B}", r"{C}")

    # now we try to find the form A, gB and tC
    ABC_forms = list()
    msepy_forms = base['forms']
    for format_form, base_rp, replace_key in zip((A, gB, tC), (base_Ar, base_Br, base_Cr), replace_keys):
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
        for _pure_lin_repr in msepy_forms:
            if _pure_lin_repr == found_root_form._pure_lin_repr:
                msepy_base_form = msepy_forms[_pure_lin_repr]
                break
            else:
                pass
        assert msepy_base_form is not None, f"we must have found a msepy copy of the root-form."
        ABC_forms.append(msepy_base_form)

    msepy_A, msepy_B, msepy_C = ABC_forms  # B is given
    nonlinear_operation = _AxBipC(*ABC_forms)

    C = nonlinear_operation(2, msepy_C, msepy_A)

    return C, msepy_B.cochain._ati_time_caller  # since B is given, its ati determine the time of C.


def _parse_trStar_rf0_dp_tr_s1_vector(dls, tr_star_rf0, tr_rf1):
    """"""
    found_root_form = None
    temp = _default_boundary_dp_vector_reprs[1].split(_sep)[2]
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
    rf1 = base['forms'][rf1._pure_lin_repr]

    b_vector_caller = _TrStarRf0DualPairingTrS1(dls, tr_star_rf0, rf1)

    return (
        MsePyDynamicLocalVector(b_vector_caller),
        b_vector_caller._time_caller,  # _TrStarRf0DualPairingTrS1 has a time caller
                                       # which determines the time of the matrix
    )


class _TrStarRf0DualPairingTrS1(Frozen):
    """"""
    def __init__(self, dls, tr_star_rf0, rf1):
        """ < tr star fr0 | trace rf1 >"""
        self._tr_star_rf0 = tr_star_rf0
        self._rf1 = rf1
        self._num_local_dofs = rf1.space.num_local_dofs(rf1._degree)
        temp = _default_boundary_dp_vector_reprs[1].split(_sep)[1]

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
            f"must provided something in ``dls.bc`` such that this b vector can be determined!"

        if _rf_evaluate_at_lin_repr in found_root_form._pure_lin_repr:

            base_form = found_root_form._pAti_form['base_form']
            self._ati = found_root_form._pAti_form['ati']
            assert all([_ is not None for _ in [base_form, self._ati]]), \
                f"we must have found a base form and its abstract time instant."

        else:
            base_form = found_root_form
            self._ati = None

        found_natural_bc = None
        for boundary_section in dls.bc:
            bcs = dls.bc[boundary_section]
            for bc in bcs:
                if bc.__class__ is MsePyDLSNaturalBoundaryCondition:

                    raw_natural_bc = bc._raw_ls_bc

                    provided_root_form = raw_natural_bc._provided_root_form

                    if self._ati is None:  # provided_root_form is the base (general) form
                        if base_form._pure_lin_repr == provided_root_form._pure_lin_repr:

                            found_natural_bc = bc
                            break

                    else:  # provided_root_form is the abstract form

                        if found_root_form._pure_lin_repr == provided_root_form._pure_lin_repr:

                            found_natural_bc = bc
                            break

                else:
                    pass

        assert found_natural_bc is not None, f"we must have found a dls natural bc!"

        self._dls_natural_bc = found_natural_bc
        msepy_base_form = None
        msepy_forms = base['forms']
        for _pure_lin_repr in msepy_forms:
            if _pure_lin_repr == base_form._pure_lin_repr:
                msepy_base_form = msepy_forms[_pure_lin_repr]
                break
            else:
                pass
        assert msepy_base_form is not None, f"we must have found a msepy root form."
        self._msepy_base_form = msepy_base_form  # may be useful in the future! just store this information here!
        self._mesh = self._msepy_base_form.mesh
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

        # now we need to make changes in _2d_data to incorporate the corresponding natural boundary condition

        nbc = self._dls_natural_bc
        # we now try to find the msepy mesh this natural bc is defined on.
        meshes = base['meshes']  # find mesh here! because only when we call it, the meshes are config-ed.
        found_msepy_mesh = None
        for mesh_sym_repr in meshes:
            msepy_mesh = meshes[mesh_sym_repr]
            msepy_manifold = msepy_mesh.manifold
            if msepy_manifold is nbc._msepy_boundary_manifold:
                found_msepy_mesh = msepy_mesh
                break
            else:
                pass
        assert found_msepy_mesh is not None, f"must found the mesh."
        assert found_msepy_mesh.__class__ is MsePyBoundarySectionMesh, \
            f"we must have found a mesh boundary section."

        # we now get the configuration of this natural bc
        category = nbc._category
        configuration = nbc._configuration

        if category == 'general_vc':   # we have one vector calculus object for all natural BC
            data = self._rf1.boundary_integrate.with_vc_over_boundary_section(
                t,
                configuration,    # the vc
                found_msepy_mesh  # over this msepy mesh boundary section.
            )

        else:
            raise NotImplementedError(f"not implemented for natural bc of category = {category}.")

        assert data.shape == (self._mesh.elements._num, self._num_local_dofs), f"array data shape wrong!"
        nbc._num_application += 1
        full_vector = MsePyStaticLocalVector(data, self._rf1.cochain.gathering_matrix)
        return full_vector
