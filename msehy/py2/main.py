# -*- coding: utf-8 -*-
r"""
"""
from src.config import SIZE   # MPI.SIZE
from src.config import _global_variables

__all__ = [

]

# will be cleared whenever new msepy implementation is called.
base = {
    'manifolds': dict(),
    'meshes': dict(),
    'spaces': dict(),  # keys: abstract space sym_repr, values: particular spaces
    'forms': dict(),  # root-forms
}


_info_cache = {
    'start_time': -1.,
    'info_count': 0,
    'info_time': -1.,
}


def _check_config():
    """Check the configuration is compatible or not."""
    assert SIZE == 1, f"meshy-py2 implementation only works for single thread call (MPI.SIZE=1), now MPI.size = {SIZE}"
    assert _global_variables['embedding_space_dim'] == 2, f"meshy-py2 implementation works only in 2d space."


def _clear_self():
    """Clear self to make sure the previous implementation does not mess things up!"""
    base['manifolds'] = dict()
    base['meshes'] = dict()
    base['spaces'] = dict()
    base['forms'] = dict()
    _info_cache['start_time'] = -1.
    _info_cache['info_count'] = 0
    _info_cache['info_time'] = -1.


def _parse_manifolds(abstract_manifolds):
    """"""
    manifold_dict = {}
    base['manifolds'] = manifold_dict


def _parse_meshes(abstract_meshes):
    """"""
    mesh_dict = {}
    base['meshes'] = mesh_dict


def _parse_spaces(abstract_spaces):
    """"""
    space_dict = {}
    base['spaces'] = space_dict


def _parse_root_forms(abstract_rfs):
    """"""
    rf_dict = {}
    base['forms'] = rf_dict


def _parse(obj):
    """The objects other than manifolds, meshes, spaces, root-forms that should be parsed for this
    particular fem setting.
    """
    return None
