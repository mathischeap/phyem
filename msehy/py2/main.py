# -*- coding: utf-8 -*-
r"""
"""
from src.config import SIZE   # MPI.SIZE
from src.config import _global_variables


__msehy_py2_setting__ = {   # can be config through ph.config.set_implementation('msehy-py2')
    'refining_examining_factor': 3,
    # To config this factor, do ``ph.config.set_implementation('msehy-py2')['refining_examining_factor'] = 4``
    'refining_examining_scheme': 0,
    # refining_examining_schemes:
    #   0)   a := int(abs(strength function)) / element_area, if a >= threshold, do refining.
}


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


from msehy.py2.manifold.main import MseHyPy2Manifold
from msehy.py2.mesh.main import MseHyPy2Mesh


def _parse_manifolds(abstract_manifolds):
    """"""
    from msepy.main import _parse_manifolds as _parse_msepy_manifolds
    _parse_msepy_manifolds(abstract_manifolds)  # implement all msepy manifold at the background.

    manifold_dict = {}
    for sym in abstract_manifolds:
        manifold = MseHyPy2Manifold(abstract_manifolds[sym])
        manifold_dict[sym] = manifold
    base['manifolds'] = manifold_dict


def _parse_meshes(abstract_meshes):
    """"""
    from msepy.main import _parse_meshes as _parse_msepy_meshes
    _parse_msepy_meshes(abstract_meshes)  # implement all msepy meshes at the background.

    mesh_dict = {}
    for sym in abstract_meshes:
        mesh = MseHyPy2Mesh(abstract_meshes[sym])
        mesh_dict[sym] = mesh

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


from msepy.main import _Config as _msepy_Config


def config(obj):
    """"""
    if obj.__class__ is MseHyPy2Manifold:  # 1: to config a msehy-py2 manifold, config its background msepy manifold
        obj = obj.background
        return _msepy_Config(obj)
    elif obj.__class__ is MseHyPy2Mesh:    # 2: to config a msehy-py2 mesh, config its background msepy mesh
        obj = obj.background
        return _msepy_Config(obj)
    else:
        raise NotImplementedError()
