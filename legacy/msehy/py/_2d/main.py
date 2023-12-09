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
    'signature': 'msehy-py2',
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


from legacy.msehy.py._2d.manifold.main import MseHyPy2Manifold
from legacy.msehy.py._2d.mesh.main import MseHyPy2Mesh
from legacy.msehy.py._2d.space.main import MseHyPy2Space
from legacy.msehy.py._2d.form.main import MseHyPy2RootForm


from msepy.main import _parse_manifolds as _parse_msepy_manifolds


def _parse_manifolds(abstract_manifolds):
    """"""
    _parse_msepy_manifolds(abstract_manifolds)  # implement all msepy manifold at the background.

    manifold_dict = {}
    for sym in abstract_manifolds:
        manifold = MseHyPy2Manifold(abstract_manifolds[sym])
        manifold_dict[sym] = manifold
    base['manifolds'] = manifold_dict


from msepy.main import _parse_meshes as _parse_msepy_meshes


def _parse_meshes(abstract_meshes):
    """"""
    _parse_msepy_meshes(abstract_meshes)  # implement all msepy meshes at the background.

    mesh_dict = {}
    for sym in abstract_meshes:
        mesh = MseHyPy2Mesh(abstract_meshes[sym])
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
                space = MseHyPy2Space(ab_sp)
                space_dict[ab_sp_sym_repr] = space
            else:
                pass
    base['spaces'] = space_dict


def _parse_root_forms(abstract_rfs):
    """"""
    rf_dict = {}
    for rf_lin_repr in abstract_rfs:  # do it for all general root-forms
        rf = abstract_rfs[rf_lin_repr]
        pure_lin_repr = rf._pure_lin_repr

        if rf._pAti_form['base_form'] is None:  # this is not a root-form at a particular time-instant.
            prf = MseHyPy2RootForm(rf)
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
            prf = MseHyPy2RootForm(rf)
            prf._pAti_form['base_form'] = particular_base_form
            prf._pAti_form['ats'] = ats
            prf._pAti_form['ati'] = ati
            rf_dict[pure_lin_repr] = prf

            assert rf_lin_repr not in particular_base_form._ats_particular_forms
            particular_base_form._ats_particular_forms[rf_lin_repr] = prf

    for pure_lin_repr in rf_dict:
        assert rf_dict[pure_lin_repr].degree is not None, f'must have degree!'

    base['forms'] = rf_dict
    base['forms'] = rf_dict


from src.wf.mp.linear_system import MatrixProxyLinearSystem


def _parse(obj):
    """The objects other than manifolds, meshes, spaces, root-forms that should be parsed for this
    particular fem setting.
    """
    if obj.__class__ is MatrixProxyLinearSystem:
        from legacy.generic.py.linear_system.localize.dynamic.main import Dynamic_Linear_System
        dynamic = Dynamic_Linear_System(obj, base)
        return dynamic

    else:
        return None  # do not raise Error (like below)!
        # raise NotImplementedError(f"cannot parse msepy implementation for {obj}.")


def _post_initialization_actions():
    """We initialize the 0th generation of space.

    Returns
    -------

    """
    pass


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
        raise NotImplementedError(
            f"msehy-py2 implementation cannot config {obj} of class: {obj.__class__.__name__}."
        )


# make new stuffs later on ...
def new(abstract_obj):
    """"""
    if abstract_obj._is_space():   #
        ab_sp_sym_repr = abstract_obj._sym_repr
        if ab_sp_sym_repr in base['spaces']:
            space = base['spaces'][ab_sp_sym_repr]
        else:
            space = MseHyPy2Space(abstract_obj)
            base['spaces'][ab_sp_sym_repr] = space
        assert abstract_obj._objective is space, f"Make sure we did not switch implementation!"
        return space
    else:
        raise NotImplementedError()


from time import time
from tools.miscellaneous.timer import MyTimer


def info(*others_2b_printed):
    """We print the info, as much as possible, of the current msepy implementation."""
    from src.time_sequence import _global_abstract_time_sequence
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
    _info_cache['info_count'] = count + 1
    _info_cache['info_time'] = new_time
    return
