# -*- coding: utf-8 -*-
r"""
"""

from tools.frozen import Frozen
from src.spaces.main import _sep
from msepy.main import base
from src.spaces.main import _str_degree_parser
from src.spaces.main import *

from src.form.main import _global_root_forms_lin_dict
from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix

from msepy.form.addons.noc.diff_3_entries_AxB_ip_C import _AxBipC, _AxBdpC
from msepy.form.addons.noc.diff_3_entries_A_convect_B_ip_C import _A_convect_B_ip_C
from msepy.form.addons.noc.diff_3_entries_dA_ip_BtpC import __dA_ip_BtpC__
from msepy.form.addons.noc.diff_3_entries_A__ip__B_tp_C import _A__ip__B_tp_C_
from msepy.form.addons.noc.AxB_ip_CxD import AxB_ip_CxD
from msepy.form.addons.noc.AxB_dp_CxD import AxB_dp_CxD

from msepy.tools.vector.dynamic import MsePyDynamicLocalVector
from msepy.tools.vector.static.local import MsePyStaticLocalVector

from msepy.tools.linear_system.dynamic.bc import MsePyDLSNaturalBoundaryCondition
from msepy.mesh.boundary_section.main import MsePyBoundarySectionMesh

from src.config import _form_evaluate_at_repr_setting
_rf_evaluate_at_lin_repr = _form_evaluate_at_repr_setting['lin']

__all__ = [
    '_indicator_check',
    '_indicator_templates',
    '_find_indicator',
    '_find_from_bracket_ABC',
    '_find_space_through_pure_lin_repr',

    '_parse_root_form',

    'Parse__M_matrix',
    "Parse__trace_matrix",  # trace matrix
    'Parse__dp_matrix',     # Wedge matrix
    'Parse__E_matrix',
    'Parse__pi_matrix',     # projection is different from (bigger than) the Hodge.
    "Parse__star_matrix",   # Hodge matrix

    'Parse__trStar_rf0_dp_tr_s1_vector',

    'Parse__astA_convect_astB_ip_tC',  # (*A .V *B, C) where AB are given and C is the test form.

    'Parse__astA_x_astB_ip_tC',   # (A x B, C) where AB are given and C is the test form.
    'Parse__astA_x_B_ip_tC',
    'Parse__A_x_astB_ip_tC',

    'Parse__astA_x_astB__dp__tC',  # <A x B | C> where AB are given and C is the test form.
    'Parse__astA_x_B__dp__tC',
    'Parse__A_x_astB__dp__tC',

    'Parse__astA_x_astB__ip__astC_x_tD',  # (A x B, C x D) where ABC are given and D is the test form.
    'Parse__A_x_astB__ip__astC_x_tD',     # (A x *B, *C x D) where BC are given and D is the test form.

    'Parse__astA_x_astB__dp__astC_x_tD',  # <A x B | C x D> where ABC are given and D is the test form.

    'Parse__IP_matrix_bf_db',

    'Parse__dastA_astA_tp_tC',
    'Parse__dastA_tB_tp_astA',
    'Parse__dtA_astB_tp_astB',
    'Parse__dA_B_tp_C__1Known',
    'Parse__dA_B_tp_C__2Known',

    'Parse__A_B_tp_C__1Known',
    'Parse__A_B_tp_C__2Known',
]


_locals = locals()

_indicator_templates = {}


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
                'symbol': symbol
            }


_indicator_cache = {}


def _find_indicator(default_setting):
    key = str(default_setting)
    if key in _indicator_cache:
        pass
    else:
        _indicator_cache[key] = default_setting[1].split(_sep)[0]
    return _indicator_cache[key]


def _find_from_bracket_ABC(default_repr, *ABC, key_words=("{A}", "{B}", "{C}")):
    """"""
    lin_reprs = default_repr[1]
    bases = lin_reprs.split(_sep)[1:]
    # now we try to find the form A, B and C
    ABC_forms = list()

    msepy_forms = base['forms']

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
        for _pure_lin_repr in msepy_forms:
            if _pure_lin_repr == found_root_form._pure_lin_repr:
                msepy_base_form = msepy_forms[_pure_lin_repr]
                break
            else:
                pass

        assert msepy_base_form is not None, f"we must have found a msepy copy of the root-form."
        ABC_forms.append(msepy_base_form)

    return ABC_forms


def _find_space_through_pure_lin_repr(_target_space_lin_repr):
    """"""
    spaces = base['spaces']
    the_msepy_space = None
    for space_lin_repr in spaces:
        msepy_space = spaces[space_lin_repr]
        abs_space_pure_lin_repr = msepy_space.abstract._pure_lin_repr
        if abs_space_pure_lin_repr == _target_space_lin_repr:
            the_msepy_space = msepy_space
            break
        else:
            pass
    assert the_msepy_space is not None, f"Find no msepy space."
    return the_msepy_space


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


def Parse__M_matrix(space, degree0, degree1):
    """"""
    degree0 = _str_degree_parser(degree0)
    degree1 = _str_degree_parser(degree1)

    the_msepy_space = _find_space_through_pure_lin_repr(space)

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


def Parse__trace_matrix(space_lin_repr, degree_str):
    """"""
    degree = _str_degree_parser(degree_str)
    space = _find_space_through_pure_lin_repr(space_lin_repr)
    gm_space_1 = space.gathering_matrix(degree)
    from src.spaces.operators import trace as space_trace
    trace_space = _find_space_through_pure_lin_repr(space_trace(space.abstract)._pure_lin_repr)
    gm_trace_0 = trace_space.gathering_matrix(degree)
    tM = space.trace_matrix(degree)
    tM = MsePyStaticLocalMatrix(
        tM,
        gm_trace_0,
        gm_space_1,
    )
    return tM, None  # time_indicator is None, mean tM is same all the time.


def Parse__dp_matrix(s0, s1, d0, d1):
    """<A|B> or <B|A>; s0 and d0 refer to the axis-0."""
    d0 = _str_degree_parser(d0)
    d1 = _str_degree_parser(d1)

    s0 = _find_space_through_pure_lin_repr(s0)
    s1 = _find_space_through_pure_lin_repr(s1)

    gm0 = s0.gathering_matrix(d0)
    gm1 = s1.gathering_matrix(d1)

    W = MsePyStaticLocalMatrix(
        s0.wedge_matrix(s1, d0, d1),
        gm0,
        gm1
    )
    return W, None  # time_indicator is None since W does not change over time.


def Parse__E_matrix(space, degree):
    """"""
    degree = _str_degree_parser(degree)
    the_msepy_space = _find_space_through_pure_lin_repr(space)
    gm0 = the_msepy_space.gathering_matrix._next(degree)
    gm1 = the_msepy_space.gathering_matrix(degree)

    E = MsePyStaticLocalMatrix(  # make a new copy every single time.
        the_msepy_space.incidence_matrix(degree),
        gm0,
        gm1,
    )

    return E, None  # time_indicator is None, mean E is same at all time.


def Parse__pi_matrix(from_space, to_space, from_d, to_d):
    """"""
    from_d = _str_degree_parser(from_d)
    to_d = _str_degree_parser(to_d)
    from_space = _find_space_through_pure_lin_repr(from_space)
    to_space = _find_space_through_pure_lin_repr(to_space)
    from msepy.space.addons.projection import MsePySpaceProjection
    pi = MsePySpaceProjection(from_space, to_space, from_d, to_d)
    return pi.matrix, None


def Parse__star_matrix(from_space, to_space, from_d, to_d):
    """Hodge matrix."""
    from_d = _str_degree_parser(from_d)
    to_d = _str_degree_parser(to_d)
    from_space = _find_space_through_pure_lin_repr(from_space)
    to_space = _find_space_through_pure_lin_repr(to_space)
    from msepy.space.addons.Hodge import MsePySpaceHodge
    H = MsePySpaceHodge(from_space, to_space, from_d, to_d)
    return H.matrix, None


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


# --- (A .V B, C) ---------------------------------------------------------------------------------
def Parse__astA_convect_astB_ip_tC(gA, gB, tC):
    """(*A .V *B, C)"""
    ABC_forms = _find_from_bracket_ABC(_VarSetting_astA_convect_astB_ip_tC, gA, gB, tC)
    _, _, msepy_C = ABC_forms
    nonlinear_operation = _A_convect_B_ip_C(*ABC_forms)
    c, time_caller = nonlinear_operation(1, msepy_C)
    return c, time_caller


# - (w x u, v) ------------------------------------------------------------------------------------
def Parse__astA_x_astB_ip_tC(gA, gB, tC):
    """(*A X *B, C), A and B are given, C is the test form, so it gives a dynamic vector."""

    ABC_forms = _find_from_bracket_ABC(_VarSetting_astA_x_astB_ip_tC, gA, gB, tC)
    _, _, msepy_C = ABC_forms
    nonlinear_operation = _AxBipC(*ABC_forms)
    c, time_caller = nonlinear_operation(1, msepy_C)
    return c, time_caller


def Parse__astA_x_B_ip_tC(gA, B, tC):
    """Remember, for this term, gA, b, tC must be root-forms."""

    ABC_forms = _find_from_bracket_ABC(_VarSetting_astA_x_B_ip_tC, gA, B, tC)
    msepy_A, msepy_B, msepy_C = ABC_forms  # A is given
    nonlinear_operation = _AxBipC(*ABC_forms)
    C = nonlinear_operation(2, msepy_C, msepy_B)

    return C, msepy_A.cochain._ati_time_caller  # since A is given, its ati determine the time of C.


def Parse__A_x_astB_ip_tC(A, gB, tC):
    """"""
    ABC_forms = _find_from_bracket_ABC(_VarSetting_A_x_astB_ip_tC, A, gB, tC)
    msepy_A, msepy_B, msepy_C = ABC_forms  # B is given
    nonlinear_operation = _AxBipC(*ABC_forms)
    C = nonlinear_operation(2, msepy_C, msepy_A)
    return C, msepy_B.cochain._ati_time_caller  # since B is given, its ati determine the time of C.


# ----- <A x B | C> -----------------------------------------------------------------------------
def Parse__astA_x_astB__dp__tC(gA, gB, tC):
    """<A X B | C>, A and B are given, C is the test form, so it gives a dynamic vector."""

    ABC_forms = _find_from_bracket_ABC(_VarSetting_astA_x_astB__dp__tC, gA, gB, tC)
    _, _, msepy_C = ABC_forms
    nonlinear_operation = _AxBdpC(*ABC_forms)
    c, time_caller = nonlinear_operation(1, msepy_C)
    return c, time_caller


def Parse__astA_x_B__dp__tC(gA, B, tC):
    """"""
    ABC_forms = _find_from_bracket_ABC(_VarSetting_astA_x_B__dp__tC, gA, B, tC)
    msepy_A, msepy_B, msepy_C = ABC_forms  # A is given
    nonlinear_operation = _AxBdpC(*ABC_forms)
    C = nonlinear_operation(2, msepy_C, msepy_B)

    return C, msepy_A.cochain._ati_time_caller  # since A is given, its ati determine the time of C.


def Parse__A_x_astB__dp__tC(A, gB, tC):
    """"""
    ABC_forms = _find_from_bracket_ABC(_VarSetting_A_x_astB__dp__tC, A, gB, tC)
    msepy_A, msepy_B, msepy_C = ABC_forms  # B is given
    nonlinear_operation = _AxBdpC(*ABC_forms)
    C = nonlinear_operation(2, msepy_C, msepy_A)
    return C, msepy_B.cochain._ati_time_caller  # since B is given, its ati determine the time of C.


# ------ (A x B, C x D) --------------------------------------------------------------------------
def Parse__astA_x_astB__ip__astC_x_tD(gA, gB, gC, tD):
    """(*A x *B, *C x D)"""
    A, B, C, D = _find_from_bracket_ABC(
        _VarSetting_astA_x_astB__ip__astC_x_tD,
        gA, gB, gC, tD,
        key_words=("{A}", "{B}", "{C}", "{D}")
    )
    nonlinear_operation = AxB_ip_CxD(A, B, C, D)
    V, time_caller = nonlinear_operation(1, D)

    return V, time_caller


def Parse__A_x_astB__ip__astC_x_tD(A, gB, gC, tD):
    """(A x *B, *C x D), D is the test function."""
    A, B, C, D = _find_from_bracket_ABC(
        _VarSetting_A_x_astB__ip__astC_x_tD,
        A, gB, gC, tD,
        key_words=("{A}", "{B}", "{C}", "{D}")
    )
    nonlinear_operation = AxB_ip_CxD(A, B, C, D)
    M, time_caller = nonlinear_operation(2, D, A)
    return M, time_caller


# ------ <A x B | C x D> --------------------------------------------------------------------------
def Parse__astA_x_astB__dp__astC_x_tD(gA, gB, gC, tD):
    """"""
    A, B, C, D = _find_from_bracket_ABC(
        _VarSetting_astA_x_astB__dp__astC_x_tD,
        gA, gB, gC, tD,
        key_words=("{A}", "{B}", "{C}", "{D}")
    )
    nonlinear_operation = AxB_dp_CxD(A, B, C, D)
    V, time_caller = nonlinear_operation(1, D)

    return V, time_caller


# - (bf, db) ------------------------------------------------------------------------------------

def Parse__IP_matrix_bf_db(sp_bf, sp_db, d_bf, d_db):
    """"""
    sp_bf = _find_space_through_pure_lin_repr(sp_bf)
    sp_db = _find_space_through_pure_lin_repr(sp_db)
    d_bf = _str_degree_parser(d_bf)
    d_db = _str_degree_parser(d_db)

    M = sp_bf[d_bf].inner_product(sp_db[d_db], special_key='(bf1, db)')

    gm_row = sp_bf.gathering_matrix(d_bf)
    gm_col = sp_db.gathering_matrix(d_db)

    M = MsePyStaticLocalMatrix(  # make a new copy every single time.
        M,
        gm_row,
        gm_col,
    )

    return M, None  # time_indicator is None, mean M is same at all time.


# (dA, B otimes C) ------------------------------------------------------------------------------------
def Parse__dastA_astA_tp_tC(gA, tC):
    """"""
    AC_forms = _find_from_bracket_ABC(_VarSetting_dastA_astA_tp_tC, gA, tC, key_words=("{A}", "{C}"))
    gA, tC = AC_forms
    nonlinear_operation = __dA_ip_BtpC__(gA, gA, tC)
    c, time_caller = nonlinear_operation(1, tC)
    return c, time_caller  # since A is given, its ati determine the time of tC.


def Parse__dastA_tB_tp_astA(gA, tB):
    """"""
    AB_forms = _find_from_bracket_ABC(_VarSetting_dastA_tB_tp_astA, gA, tB, key_words=("{A}", "{B}"))
    gA, tB = AB_forms
    nonlinear_operation = __dA_ip_BtpC__(gA, tB, gA)
    b, time_caller = nonlinear_operation(1, tB)
    return b, time_caller  # since A is given, its ati determine the time of tB.


def Parse__dtA_astB_tp_astB(tA, gB):
    """"""
    AB_forms = _find_from_bracket_ABC(_VarSetting_dtA_astB_tp_astB, tA, gB, key_words=("{A}", "{B}"))
    tA, gB = AB_forms
    nonlinear_operation = __dA_ip_BtpC__(tA, gB, gB)
    a, time_caller = nonlinear_operation(1, tA)
    return a, time_caller  # since B is given, its ati determine the time of tA.


def Parse__dA_B_tp_C__1Known(A, B, C, kf, tf, uk):
    """"""
    forms = _find_from_bracket_ABC(
        _VarSetting_dA_B_tp_C__1Known,
        A, B, C, kf, tf, uk,
        key_words=("{A}", "{B}", "{C}", "{K}", "{T}", "{U}")
    )
    A, B, C, kf, tf, uk = forms
    nonlinear_operation = __dA_ip_BtpC__(A, B, C)
    M = nonlinear_operation(2, tf, uk)
    return M, kf.cochain._ati_time_caller


def Parse__dA_B_tp_C__2Known(A, B, C, kf1, kf2, tf):
    """"""
    forms = _find_from_bracket_ABC(
        _VarSetting_dA_B_tp_C__2Known,
        A, B, C, kf1, kf2, tf,
        key_words=("{A}", "{B}", "{C}", "{K1}", "{K2}", "{T}")
    )
    A, B, C, kf1, kf2, tf = forms
    nonlinear_operation = __dA_ip_BtpC__(A, B, C)
    V, time_caller = nonlinear_operation(1, tf)
    return V, time_caller


# (A, B otimes C) ------------------------------------------------------------------------------------
def Parse__A_B_tp_C__1Known(A, B, C, kf, tf, uk):
    """"""
    forms = _find_from_bracket_ABC(
        _VarSetting_A_B_tp_C__1Known,
        A, B, C, kf, tf, uk,
        key_words=("{A}", "{B}", "{C}", "{K}", "{T}", "{U}")
    )
    A, B, C, kf, tf, uk = forms
    nonlinear_operation = _A__ip__B_tp_C_(A, B, C)
    M = nonlinear_operation(2, tf, uk)
    return M, kf.cochain._ati_time_caller


def Parse__A_B_tp_C__2Known(A, B, C, kf1, kf2, tf):
    """"""
    forms = _find_from_bracket_ABC(
        _VarSetting_A_B_tp_C__2Known,
        A, B, C, kf1, kf2, tf,
        key_words=("{A}", "{B}", "{C}", "{K1}", "{K2}", "{T}")
    )
    A, B, C, kf1, kf2, tf = forms
    nonlinear_operation = _A__ip__B_tp_C_(A, B, C)
    V, time_caller = nonlinear_operation(1, tf)
    return V, time_caller
