# -*- coding: utf-8 -*-
r"""
"""
from src.config import RANK, MASTER_RANK, COMM
from src.config import _global_variables


# will be cleared whenever new msepy implementation is called.
base = {
    'signature': 'mpi-msehy-py2',
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
    assert COMM is not None, f"must enable mpi4py for mpi-msehy-py2 implementation."
    COMM.barrier()  # sync here!
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


from _MPI.msehy.py._2d.manifold.main import MPI_MseHy_Py2_Manifold
from _MPI.msehy.py._2d.mesh.main import MPI_MseHy_Py2_Mesh
from _MPI.msehy.py._2d.space.main import MPI_MseHy_Py2_Space
from _MPI.msehy.py._2d.form.main import MPI_MseHy_Py2_Form


from msepy.main import _parse_manifolds as _parse_msepy_manifolds


def _parse_manifolds(abstract_manifolds):
    """"""
    if RANK == MASTER_RANK:
        _parse_msepy_manifolds(abstract_manifolds)  # implement all msepy manifold at the background.
    else:
        pass

    manifold_dict = {}
    for sym in abstract_manifolds:
        manifold = MPI_MseHy_Py2_Manifold(abstract_manifolds[sym])
        manifold_dict[sym] = manifold
    base['manifolds'] = manifold_dict


from msepy.main import _parse_meshes as _parse_msepy_meshes
from msehy.py._2d.main import _parse_meshes as _parse_msehy_meshes


def _parse_meshes(abstract_meshes):
    """"""
    if RANK == MASTER_RANK:
        _parse_msepy_meshes(abstract_meshes)  # implement all msepy meshes at the background in the master rank.
        _parse_msehy_meshes(abstract_meshes)
    else:
        pass

    mesh_dict = {}
    for sym in abstract_meshes:
        mesh = MPI_MseHy_Py2_Mesh(abstract_meshes[sym])
        mesh_dict[sym] = mesh
    base['meshes'] = mesh_dict


def _parse_spaces(abstract_spaces):
    """"""
    space_dict = {}
    for ab_msh_sym_repr in abstract_spaces:
        ab_sps = abstract_spaces[ab_msh_sym_repr]

        for ab_sp_sym_repr in ab_sps:
            ab_sp = ab_sps[ab_sp_sym_repr]

            if ab_sp.orientation != 'unknown':  # Those spaces are probably not for root-forms, skipping is OK.
                space = MPI_MseHy_Py2_Space(ab_sp)
                space_dict[ab_sp_sym_repr] = space
            else:
                pass
    base['spaces'] = space_dict


def _parse_root_forms(abstract_rfs):
    """"""


def _parse(obj):
    """The objects other than manifolds, meshes, spaces, root-forms that should be parsed for this
    particular fem setting.
    """
    return None  # do not raise Error (like below)!
    # raise NotImplementedError(f"cannot parse msepy implementation for {obj}.")


def _post_initialization_actions():
    """

    Returns
    -------

    """
    pass


from msepy.main import _Config as _msepy_Config
from tools.void import VoidClass


def config(obj):
    """"""
    if obj.__class__ is MPI_MseHy_Py2_Manifold:
        # 1: to config a mpi-msehy-py2 manifold, config its background msepy manifold
        if RANK == MASTER_RANK:
            obj = obj.background
            return _msepy_Config(obj)
        else:
            return VoidClass()
    elif obj.__class__ is MPI_MseHy_Py2_Mesh:
        # 2: to config a mpi-msehy-py2 mesh, config its background msepy mesh
        if RANK == MASTER_RANK:
            obj = obj.background.background  # -> msehy mesh -> msepy mesh
            return _msepy_Config(obj)
        else:
            return VoidClass()
    else:
        raise NotImplementedError(
            f"msehy-py2 implementation cannot config {obj} of class: {obj.__class__.__name__}."
        )


from time import time
from tools.miscellaneous.timer import MyTimer


def info(*others_2b_printed):
    """We print the info, as much as possible, of the current msepy implementation."""
    forms = base['forms']
    # -- first we print the newest time of the cochain (if there is) of each form.
    count = _info_cache['info_count']
    old_time = _info_cache['info_time']
    if _info_cache['start_time'] == -1.:
        _info_cache['start_time'] = time()
    else:
        pass
    start_time = _info_cache['start_time']
    if old_time == -1:
        old_time = time()
    else:
        pass
    new_time = time()
    total_cost = new_time - start_time
    if RANK == MASTER_RANK:
        from src.time_sequence import _global_abstract_time_sequence
        print(f'=== [{count}] {MyTimer.current_time()} -after- %.2f(s),'
              f' total: {MyTimer.seconds2dhms(total_cost)} <----' % (new_time - old_time))
        print(f"~) Form with newest cochain @ ---------------------------- ")
        for form_sym in forms:
            form = forms[form_sym]
            if form._is_base():
                newest_time = form.cochain.newest
                if newest_time is not None:
                    print('{:>20} @ {:<15}'.format(
                        form.abstract._pure_lin_repr, newest_time
                    ))
            else:
                pass
        print(f"\n~) Existing time sequences ------------------------------- ")
        for ats_lin in _global_abstract_time_sequence:
            ats = _global_abstract_time_sequence[ats_lin]
            ats.info()

        print(f"\n~) Others: ~~~~")
        for i, other in enumerate(others_2b_printed):
            print(f"  {i}) -> {other}\n")

        print()
    else:
        pass

    _info_cache['info_count'] = count + 1
    _info_cache['info_time'] = new_time
    return
