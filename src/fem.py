# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
r"""
To bring an abstract setting to an implementation, we just need to call ``apply`` in ``fem`` module of *phyem*,
for example,

.. code-block::

    implementation, objects = ph.fem.apply(arg0, arg1)

This function takes two arguments,

- the first argument, ``arg0``, is an indicator implying which implementation we want to use,
- the second argument, ``arg1``, is a dictionary of target objects; if we use ``local()``,
  we target at all local variables.

.. important::

    The implemented implementations are

    +-------------------------+-----------------------------------------------------------------+--------------------+
    | **indicator**           | **description**                                                 | **parallelizable** |
    +-------------------------+-----------------------------------------------------------------+--------------------+
    | ``'msepy'``             | mimetic spectral elements in Python                             | No                 |
    +-------------------------+-----------------------------------------------------------------+--------------------+

The outputs are the implementation body, i.e. ``implementation``, and a dictionary of all instances that have
their counterparts in this implementation, i.e. ``objects``, whose keys are the abstract variable names and
whose values are the counterparts.

If an abstract instance has no counterpart in the specified implementation, but it is still
sent to ``apply`` (for example, through global variable dictionary ``local()``), it returns
no error (will just be ignored).

"""

from phyem.src.manifold import Manifold
from phyem.src.mesh import Mesh
from phyem.src.spaces.base import SpaceBase
from phyem.src.form.main import Form

from phyem.src.manifold import _global_manifolds              # [manifold_sym_repr] -> manifold
from phyem.src.mesh import _global_meshes                     # [mesh_sym_repr] -> mesh
from phyem.src.spaces.main import _space_set                  # [mesh_sym_repr][space_sym_repr] -> space
from phyem.src.form.main import _global_root_forms_lin_dict   # [root-form_lin_repr] -> root-form

# implemented implementations ------------------------------------------------------------------------------
import phyem.msepy.main as msepy                                # mimetic spectral elements, python implementation

import phyem.msehtt.static.main as msehtt_static                # static version of msehtt
import phyem.msehtt.adaptive.main as msehtt_adaptive            # adaptive version of msehtt

# import msehtt_ncf.static.main as msehtt_ncf_static        # static version of msehtt-ncf
# ==========================================================================================================

_implemented_finite_elements = {
    'msepy': msepy,                  # mimetic spectral elements, python implementation

    'msehtt':        msehtt_static,  # default version of msehtt is the static one.
    'msehtt-s':      msehtt_static,  # shortcut of msehtt-static
    'msehtt-static': msehtt_static,  # static version of msehtt

    'msehtt-a':        msehtt_adaptive,
    'msehtt-adaptive': msehtt_adaptive,

    # 'msehtt-ncf':        msehtt_ncf_static,  # default version of msehtt-ncf is the static one
    # 'msehtt-ncf-s':      msehtt_ncf_static,  # shortcut of static version of msehtt-ncf
    # 'msehtt-ncf-static': msehtt_ncf_static,  # static version of msehtt-ncf
}


def apply(fe_name, obj_dict):
    r"""

    Parameters
    ----------
    fe_name
    obj_dict

    Returns
    -------

    """
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

    if hasattr(implementation, '_post_initialization_actions'):
        implementation._post_initialization_actions()
    else:
        pass

    return implementation, obj_space


def _parse_obj(implementation, obj):
    r"""

    Parameters
    ----------
    implementation
    obj

    Returns
    -------

    """
    if obj.__class__ is Manifold:
        return implementation.base['manifolds'][obj._sym_repr]
    elif obj.__class__ is Mesh:
        return implementation.base['meshes'][obj._sym_repr]
    elif issubclass(obj.__class__, SpaceBase):
        if obj._sym_repr in implementation.base['spaces']:
            return implementation.base['spaces'][obj._sym_repr]
        else:
            return None  # for those spaces have no particular counterparts, we simply skip them.
    elif obj.__class__ is Form:
        if obj.is_root():
            if obj._pure_lin_repr in implementation.base['forms']:
                return implementation.base['forms'][obj._pure_lin_repr]
            else:
                return None  # for those spaces have no particular counterparts, we simply skip them.
        else:
            return None  # non-root-form has no counterpart.
    else:
        return implementation._parse(obj)
