# -*- coding: utf-8 -*-
r"""
finite element setting
"""
from src.manifold import Manifold
from src.mesh import Mesh
from src.spaces.base import SpaceBase
from src.form.main import Form

from src.manifold import _global_manifolds  # [manifold_sym_repr] -> manifold
from src.mesh import _global_meshes  # [mesh_sym_repr] -> mesh
from src.spaces.main import _space_set  # [mesh_sym_repr][space_sym_repr] -> space
from src.form.main import _global_root_forms_lin_dict  # [root-form_lin_repr] -> root-form
import msepy.main as msepy
import msehy.py2.main as msehy_py2


_implemented_finite_elements = {
    'msepy': msepy,   # mimetic spectral elements, python implementation
    'msehy-py2': msehy_py2,  # hybrid mimetic spectral elements , python implementation , 2-dimensions.
}


def apply(fe_name, obj_dict):
    """"""
    assert fe_name in _implemented_finite_elements, \
        f"finite element name={fe_name} is wrong, should be one of {_implemented_finite_elements.keys()}"

    implementation = _implemented_finite_elements[fe_name]
    implementation._check_config()                      # check the configuration.
    implementation._clear_self()                        # make sure we clear everything from previous implementation
    implementation._parse_manifolds(_global_manifolds)  # important, for all manifolds
    implementation._parse_meshes(_global_meshes)        # important, for all meshes
    implementation._parse_spaces(_space_set)            # important, for all spaces
    implementation._parse_root_forms(_global_root_forms_lin_dict)   # important, for all root-forms

    obj_space = dict()
    for obj_name in obj_dict:
        obj = obj_dict[obj_name]
        particular_obj = _parse_obj(implementation, obj)
        if particular_obj is not None:
            obj_space[obj_name] = particular_obj
        else:
            pass

    return implementation, obj_space


def _parse_obj(implementation, obj):
    """"""
    if obj.__class__ is Manifold:
        return implementation.base['manifolds'][obj._sym_repr]
    elif obj.__class__ is Mesh:
        return implementation.base['meshes'][obj._sym_repr]
    elif issubclass(obj.__class__, SpaceBase):
        if obj._sym_repr in implementation.base['spaces']:
            return implementation.base['spaces'][obj._sym_repr]
        else:
            pass  # for those spaces have no particular counterparts, we simply skip them.
    elif obj.__class__ is Form:
        if obj.is_root():
            if obj._pure_lin_repr in implementation.base['forms']:
                return implementation.base['forms'][obj._pure_lin_repr]
            else:
                pass  # for those spaces have no particular counterparts, we simply skip them.
        else:
            pass  # non-root-form has no counterpart.
    else:
        return implementation._parse(obj)
