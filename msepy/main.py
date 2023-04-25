# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
created at: 3/30/2023 6:19 PM
"""
import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.manifold.main import MsePyManifold
from msepy.mesh.main import MsePyMesh
from msepy.space.main import MsePySpace
from msepy.form.main import MsePyRootForm
from src.config import SIZE   # MPI.SIZE


__all__ = [
    '_parse_manifolds',
    '_parse_meshes',
    '_parse_spaces',
    '_parse_root_forms',
    '_parse',

    'config',
]


base = {
    'manifolds': dict(),
    'meshes': dict(),
    'spaces': dict(),
    'forms': dict(),  # root-forms
}


def _check_config():
    """"""
    assert SIZE == 1, f"msepy only works for single thread call (MPI.SIZE=1), now MPI.size = {SIZE}"


def _parse_manifolds(abstract_manifolds):
    """"""
    manifold_dict = {}
    for sym in abstract_manifolds:
        manifold = MsePyManifold(abstract_manifolds[sym])
        manifold_dict[sym] = manifold
    base['manifolds'] = manifold_dict


def _parse_meshes(abstract_meshes):
    """"""
    mesh_dict = {}
    for sym in abstract_meshes:
        am = abstract_meshes[sym]
        m = MsePyMesh(am)
        mesh_dict[sym] = m
    base['meshes'] = mesh_dict


def _parse_spaces(abstract_spaces):
    """"""
    space_dict = {}
    for ab_msh_sym_repr in abstract_spaces:
        ab_sps = abstract_spaces[ab_msh_sym_repr]

        for ab_sp_sym_repr in ab_sps:
            ab_sp = ab_sps[ab_sp_sym_repr]

            if ab_sp.orientation != 'unknown':  # Those spaces are probably not for root-forms, skipping is OK.
                space = MsePySpace(ab_sp)
                space_dict[ab_sp_sym_repr] = space
            else:
                pass
    base['spaces'] = space_dict


def _parse_root_forms(abstract_rfs):
    """"""
    rf_dict = {}
    for rf_lin_repr in abstract_rfs:  # do it for all general root-forms
        rf = abstract_rfs[rf_lin_repr]

        if rf._pAti_form['base_form'] is None:  # this is not a root-form at a particular time-instant.
            prf = MsePyRootForm(rf)
            rf_dict[rf_lin_repr] = prf
        else:
            pass

    for rf_lin_repr in abstract_rfs:  # then do it for all root-forms at particular time instant
        rf = abstract_rfs[rf_lin_repr]
        if rf._pAti_form['base_form'] is None:
            pass
        else:
            base_form = rf._pAti_form['base_form']
            ats = rf._pAti_form['ats']
            ati = rf._pAti_form['ati']

            particular_base_form = rf_dict[base_form._lin_repr]
            prf = MsePyRootForm(rf)
            prf._pAti_form['base_form'] = particular_base_form
            prf._pAti_form['ats'] = ats
            prf._pAti_form['ati'] = ati
            rf_dict[rf_lin_repr] = prf

            assert rf_lin_repr not in particular_base_form._ats_particular_forms
            particular_base_form._ats_particular_forms[rf_lin_repr] = prf
    for rf_lin_repr in rf_dict:
        assert rf_dict[rf_lin_repr].degree is not None
    base['forms'] = rf_dict


def _parse(obj):
    """The objects other than manifolds, meshes, spaces, root-forms that should be parsed for this
    particular fem setting.
    """


from msepy.manifold.main import config as _mf_config
from msepy.mesh.main import config as _mh_config


def config(obj):
    return _Config(obj)


class _Config(Frozen):
    """"""
    def __init__(self, obj):
        self._obj = obj

    def __call__(self, *args, **kwargs):
        if self._obj.__class__ is MsePyManifold:
            return _mf_config(self._obj, *args, **kwargs)
        elif self._obj.__class__ is MsePyMesh:
            mesh = self._obj
            abstract_mesh = mesh.abstract
            abstract_manifold = abstract_mesh.manifold
            mnf = None
            for mnf_sr in base['manifolds']:
                mnf = base['manifolds'][mnf_sr]
                if mnf.abstract is abstract_manifold:
                    break
            assert mnf is not None, f"cannot find a valid mse-py-manifold."

            return _mh_config(self._obj, mnf, *args, **kwargs)
        else:
            raise NotImplementedError()
