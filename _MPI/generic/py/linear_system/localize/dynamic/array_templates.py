# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.spaces.main import _sep
from src.spaces.main import *
from src.form.main import _global_root_forms_lin_dict

from _MPI.generic.py._2d_unstruct.form.main import MPI_Py_2D_Unstructured_Form
from _MPI.generic.py.vector.localize.static import MPI_PY_Localize_Static_Vector

from _MPI.generic.py.tools.nolinear_operation.AxB_ip_C import MPI_PY_AxBipC

from src.config import _form_evaluate_at_repr_setting
_rf_evaluate_at_lin_repr = _form_evaluate_at_repr_setting['lin']

__all__ = [
    '_indicator_check',
    '_indicator_templates',
    '_find_indicator',
    '_find_py_space_through_pure_lin_repr',

    '_parse_root_form',

    'Parse__M_matrix',
    'Parse__E_matrix',

    'Parse__MPI_PY_trStar_rf0_dp_tr_s1_vector',

    'Parse__astA_x_astB_ip_tC',
    'Parse__astA_x_B_ip_tC',
    'Parse__A_x_astB_ip_tC',
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


from generic.py.linear_system.localize.dynamic.array_templates import _find_py_space_through_pure_lin_repr


def _find_from_bracket_ABC(base, default_repr, *ABC, key_words=("{A}", "{B}", "{C}")):
    """"""
    lin_reprs = default_repr[1]
    bases = lin_reprs.split(_sep)[1:]
    # now we try to find the form gA, B and tC
    ABC_forms = list()

    forms = base['forms']

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

        the_form = None
        for _pure_lin_repr in forms:
            if _pure_lin_repr == found_root_form._pure_lin_repr:
                the_form = forms[_pure_lin_repr]
                break
            else:
                pass
        assert the_form is not None, f"we must have found a msepy copy of the root-form."

        if the_form is not MPI_Py_2D_Unstructured_Form:

            if hasattr(the_form, 'generic'):
                the_form = the_form.generic
            else:
                raise Exception()
        else:
            pass

        ABC_forms.append(the_form)

    return ABC_forms


# ====================================================================================================
from generic.py.linear_system.localize.dynamic.array_templates import _parse_root_form
from generic.py.linear_system.localize.dynamic.array_templates import Parse__M_matrix
from generic.py.linear_system.localize.dynamic.array_templates import Parse__E_matrix


# --------- natural bc -----------------------------------------------------------------------------

def Parse__MPI_PY_trStar_rf0_dp_tr_s1_vector(dls, tr_star_rf0, tr_rf1):
    """Natural boundary condition term vector maker."""
    found_test_root_form = None
    temp = _VarSetting_boundary_dp_vector[1].split(_sep)[2]
    for root_form_lin_repr in _global_root_forms_lin_dict:
        check_form = _global_root_forms_lin_dict[root_form_lin_repr]
        check_temp = temp.replace('{f1}', check_form._pure_lin_repr)
        if check_temp == tr_rf1:
            found_test_root_form = check_form
            break
        else:
            pass
    assert found_test_root_form is not None, f"we must have found a root form!"

    rf1 = found_test_root_form
    rf1 = dls._base['forms'][rf1._pure_lin_repr]

    b_vector_caller = _MPI_PY_TrStarRf0DualPairingTrS1(dls, tr_star_rf0, rf1)

    from _MPI.generic.py.vector.localize.dynamic import MPI_PY_Localize_Dynamic_Vector

    return (
        MPI_PY_Localize_Dynamic_Vector(b_vector_caller),
        b_vector_caller._time_caller,  # _TrStarRf0DualPairingTrS1 has a time caller
        # which determines the time of the matrix
    )


class _MPI_PY_TrStarRf0DualPairingTrS1(Frozen):
    """"""
    def __init__(self, dls, tr_star_rf0, rf1):
        """ < tr star fr0 | trace rf1 >"""
        self._dls = dls

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
            f"must provided something in ``dls.bc`` such that this b vector can be determined!"

        if _rf_evaluate_at_lin_repr in found_root_form._pure_lin_repr:

            base_form_f0 = found_root_form._pAti_form['base_form']
            self._ati = found_root_form._pAti_form['ati']
            assert all([_ is not None for _ in [base_form_f0, self._ati]]), \
                f"we must have found a base form and its abstract time instant."

        else:
            base_form_f0 = found_root_form
            self._ati = None

        # base_form_f0 represent rf0 in the particular implementation

        from generic.py.linear_system.localize.dynamic.bc import Natural_BoundaryCondition
        found_natural_bc = None
        for boundary_section in dls.bc:
            bcs = dls.bc[boundary_section]
            for bc in bcs:
                if bc.__class__ is Natural_BoundaryCondition:

                    raw_natural_bc = bc._raw_ls_bc

                    provided_root_form = raw_natural_bc._provided_root_form

                    if self._ati is None:  # provided_root_form is the base (general) form
                        if base_form_f0._pure_lin_repr == provided_root_form._pure_lin_repr:

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
        implementation_base_form = None
        msepy_forms = dls._base['forms']
        for _pure_lin_repr in msepy_forms:
            if _pure_lin_repr == base_form_f0._pure_lin_repr:
                implementation_base_form = msepy_forms[_pure_lin_repr]
                break
            else:
                pass
        assert implementation_base_form is not None, f"we must have found a msepy root form."
        self._implementation_base_form = implementation_base_form
        # may be useful in the future! just store this information here!
        self._mesh = self._implementation_base_form.mesh
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
        meshes = self._dls._base['meshes']  # find mesh here! because only when we call it, the meshes are config-ed.
        found_boundary_mesh = None
        for mesh_sym_repr in meshes:
            mesh = meshes[mesh_sym_repr]
            manifold = mesh.manifold
            if manifold is nbc._boundary_manifold:
                found_boundary_mesh = mesh
                break
            else:
                pass
        assert found_boundary_mesh is not None, f"must found the mesh."

        # ------------ for different implementations ---------------------------------------
        from _MPI.msehy.py._2d.mesh.main import MPI_MseHy_Py2_Mesh
        if found_boundary_mesh.__class__ is MPI_MseHy_Py2_Mesh:
            assert not found_boundary_mesh._is_mesh(), f"must represent a boundary section"
            found_boundary_mesh = found_boundary_mesh.generic
        else:
            raise NotImplementedError()
        # ==================================================================================

        configuration = nbc._configuration   # the boundary condition: tr-star-rf0
        data = self._rf1.boundary_integrate.with_vc_over_boundary_section(
            t,
            configuration,    # the configuration; usually a vc.
            found_boundary_mesh  # over this msepy mesh boundary section.
        )
        assert isinstance(data, dict), f"static vector data must be in a dictionary."
        nbc._num_application += 1
        full_vector = MPI_PY_Localize_Static_Vector(data, self._rf1.cochain.gathering_matrix)
        return full_vector


# ------- (w x u, v) -------------------------------------------------------------------------------
def Parse__astA_x_astB_ip_tC(base, gA, gB, tC):
    """(A X B, C), A and B are given, C is the test form, so it gives a dynamic vector."""
    ABC_forms = _find_from_bracket_ABC(base, _VarSetting_astA_x_astB_ip_tC, gA, gB, tC)
    _, _, msepy_C = ABC_forms
    nonlinear_operation = MPI_PY_AxBipC(*ABC_forms)
    c, time_caller = nonlinear_operation(1, msepy_C)
    return c, time_caller  # since A is given, its ati determine the time of C.


def Parse__astA_x_B_ip_tC(base, gA, B, tC):
    """Remember, for this term, gA, b, tC must be root-forms."""
    ABC_forms = _find_from_bracket_ABC(base, _VarSetting_astA_x_B_ip_tC, gA, B, tC)
    msepy_A, msepy_B, msepy_C = ABC_forms  # A is given
    nonlinear_operation = MPI_PY_AxBipC(*ABC_forms)
    C = nonlinear_operation(2, msepy_C, msepy_B)
    return C, msepy_A.cochain._ati_time_caller  # since A is given, its ati determine the time of C.


def Parse__A_x_astB_ip_tC(base, A, gB, tC):
    """"""
    ABC_forms = _find_from_bracket_ABC(base, _VarSetting_A_x_astB_ip_tC, A, gB, tC)
    msepy_A, msepy_B, msepy_C = ABC_forms  # B is given
    nonlinear_operation = MPI_PY_AxBipC(*ABC_forms)
    C = nonlinear_operation(2, msepy_C, msepy_A)
    return C, msepy_B.cochain._ati_time_caller  # since B is given, its ati determine the time of C.
