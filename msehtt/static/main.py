# -*- coding: utf-8 -*-
r"""
"""
base = {
    'manifolds': dict(),  # keys: abstract manifold sym_repr
    'meshes': dict(),     # keys: abstract mesh sym_repr
    'spaces': dict(),  # keys: abstract space sym_repr, values: MsePy spaces
    'forms': dict(),   # keys: abstract root form pure_lin_repr, values: root-forms,
    'the_great_mesh': None,   # all forms/meshes will also point to this great mesh.
    'PARSER': None,           # the parser that will study this base
}


__all__ = [
    'PARSER',
]


import msehtt.static.implementation_array_parser as PARSER
PARSER._setting_['base'] = base
base['PARSER'] = PARSER


from msehtt.static.manifold.main import MseHttManifold
from msehtt.static.mesh.partial.main import MseHttMeshPartial
from msehtt.static.mesh.great.main import MseHttGreatMesh
from msehtt.static.space.main import MseHttSpace
from msehtt.static.form.main import MseHttForm

from src.wf.mp.linear_system import MatrixProxyLinearSystem
from src.wf.mp.nonlinear_system import MatrixProxyNoneLinearSystem


def _check_config():
    """Check whether the configuration is compatible or not. And if necessary, prepare the data!"""


def _clear_self():
    """Clear self to make sure the previous implementation does not mess things up!"""
    base['manifolds'] = dict()
    base['meshes'] = dict()
    base['spaces'] = dict()
    base['forms'] = dict()
    base['the_great_mesh'] = None


def _parse_manifolds(abstract_manifolds):
    """"""
    manifold_dict = {}
    for sym in abstract_manifolds:
        manifold = MseHttManifold(abstract_manifolds[sym])
        manifold_dict[sym] = manifold
    base['manifolds'] = manifold_dict


def tgm():
    """return the great mesh."""
    return base['the_great_mesh']


def _parse_meshes(abstract_meshes):
    """"""
    mesh_dict = {}
    for sym in abstract_meshes:
        am = abstract_meshes[sym]
        pm = MseHttMeshPartial(am)
        mesh_dict[sym] = pm
    base['meshes'] = mesh_dict
    assert base['the_great_mesh'] is None, f"We must do not generate the great mesh yet."
    # noinspection PyTypedDict
    base['the_great_mesh'] = MseHttGreatMesh()


def _parse_spaces(abstract_spaces):
    """"""
    space_dict = {}
    for ab_msh_sym_repr in abstract_spaces:
        ab_sps = abstract_spaces[ab_msh_sym_repr]

        for ab_sp_sym_repr in ab_sps:
            ab_sp = ab_sps[ab_sp_sym_repr]

            if ab_sp.orientation == 'unknown':
                # These spaces are probably not for root-forms, skipping is OK.
                pass
            else:
                space = MseHttSpace(ab_sp)
                space_dict[ab_sp_sym_repr] = space

                the_msehtt_partial_mesh = None
                for mesh_repr in base['meshes']:
                    if mesh_repr == space.abstract.mesh._sym_repr:
                        the_msehtt_partial_mesh = base['meshes'][mesh_repr]
                        break
                    else:
                        pass

                assert the_msehtt_partial_mesh is not None, f"we must have found a msehtt partial mesh!"
                space._tpm = the_msehtt_partial_mesh

    base['spaces'] = space_dict


def _parse_root_forms(abstract_rfs):
    """"""
    rf_dict = {}
    for rf_lin_repr in abstract_rfs:  # do it for all general root-forms
        rf = abstract_rfs[rf_lin_repr]
        pure_lin_repr = rf._pure_lin_repr

        if rf._pAti_form['base_form'] is None:  # this is not a root-form at a particular time-instant.
            prf = MseHttForm(rf)
            rf_dict[pure_lin_repr] = prf
        else:
            pass

    for rf_lin_repr in abstract_rfs:  # then do it for all root-forms at particular time instant
        rf = abstract_rfs[rf_lin_repr]
        pure_lin_repr = rf._pure_lin_repr

        if rf._pAti_form['base_form'] is None:
            pass
        else:  # this is a root-form at a particular time-instant.
            base_form = rf._pAti_form['base_form']
            ats = rf._pAti_form['ats']
            ati = rf._pAti_form['ati']

            particular_base_form = rf_dict[base_form._pure_lin_repr]
            prf = MseHttForm(rf)
            prf._pAti_form['base_form'] = particular_base_form
            prf._pAti_form['ats'] = ats
            prf._pAti_form['ati'] = ati
            rf_dict[pure_lin_repr] = prf

            assert rf_lin_repr not in particular_base_form._ats_particular_forms
            particular_base_form._ats_particular_forms[rf_lin_repr] = prf

    for pure_lin_repr in rf_dict:
        form = rf_dict[pure_lin_repr]
        assert form.degree is not None, \
            f"msehtt form must have a degree, an abstract root of no degree cannot be implemented."

    base['forms'] = rf_dict
    ___link_all_forms____()


def ___link_all_forms____():
    """"""
    the_great_mesh = base['the_great_mesh']
    partial_meshes = base['meshes']  # all the partial meshes
    spaces = base['spaces']          # all the msehtt spaces
    manifolds = base['manifolds']     # all the msehtt manifolds
    forms = base['forms']            # all the forms

    for f_sym in forms:
        form = forms[f_sym]
        abstract_form = form.abstract
        abstract_space = abstract_form.space._sym_repr
        abstract_mesh = abstract_form.mesh._sym_repr
        abstract_manifold = abstract_form.manifold._sym_repr

        form._tgm = the_great_mesh
        form._tpm = partial_meshes[abstract_mesh]
        form._manifold = manifolds[abstract_manifold]
        form._space = spaces[abstract_space]


def _parse(obj):
    """The objects other than manifolds, meshes, spaces, root-forms that should be parsed for this
    particular fem setting.
    """
    if obj.__class__ is MatrixProxyLinearSystem:
        from msehtt.tools.linear_system.dynamic.main import MseHttDynamicLinearSystem
        dynamic = MseHttDynamicLinearSystem(obj, base)
        return dynamic
    elif obj.__class__ is MatrixProxyNoneLinearSystem:
        from msehtt.tools.nonlinear_system.dynamic.main import MseHttDynamicNonLinearSystem
        dynamic = MseHttDynamicNonLinearSystem(obj, base)
        return dynamic

    else:
        return None  # do not raise Error (like below)!
        # raise NotImplementedError(f"cannot parse msepy implementation for {obj}.")


class config:
    def __init__(self, obj):
        self._obj = obj

    def __call__(self, *args, **kwargs):
        _config(self._obj)(*args, **kwargs)


# noinspection PyUnresolvedReferences
def _config(obj=None):
    """"""
    if obj is None or (isinstance(obj, str) and obj == 'tgm'):  # we are configuring the great mesh.
        obj = base['the_great_mesh']
    else:
        pass

    return obj._config


from msehtt.tools.gathering_matrix import ___clean_cache_msehtt_gm___
from msehtt.static.mesh.great.elements.types.base import ___clean_cache_msehtt_element_ct___


def clean_cache():
    """"""
    ___clean_cache_msehtt_gm___()
    ___clean_cache_msehtt_element_ct___()
