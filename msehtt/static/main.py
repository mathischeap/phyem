# -*- coding: utf-8 -*-
r"""
"""
base = {
    'manifolds': dict(),  # keys: abstract manifold sym_repr
    'meshes': dict(),     # keys: abstract mesh sym_repr
    'spaces': dict(),  # keys: abstract space sym_repr, values: MsePy spaces
    'forms': dict(),   # keys: abstract root form pure_lin_repr, values: root-forms,
    'the_great_mesh': None,   # all forms/meshes will also point to this great mesh.
    'PARSER': None,           # the linear parser that will study this base
    'NOC-PARSER': None,       # the nonlinear parser that will study this base
}


__all__ = [
    'PARSER',
]


import phyem.msehtt.static.implementation_array_parser as PARSER
PARSER._setting_['base'] = base
# noinspection PyTypeChecker
base['PARSER'] = PARSER

import phyem.msehtt.static.implementation_nop_parser as NOC_PARSER
NOC_PARSER._setting_['base'] = base
# noinspection PyTypeChecker
base['NOC-PARSER'] = NOC_PARSER

from phyem.msehtt.static.manifold.main import MseHttManifold
from phyem.msehtt.static.mesh.partial.main import MseHttMeshPartial
from phyem.msehtt.static.mesh.great.main import MseHttGreatMesh
from phyem.msehtt.static.space.main import MseHttSpace
from phyem.msehtt.static.form.main import MseHttForm

from phyem.src.wf.mp.linear_system import MatrixProxyLinearSystem
from phyem.src.wf.mp.nonlinear_system import MatrixProxyNoneLinearSystem


from phyem.src.config import RANK, MASTER_RANK


def _check_config():
    """Check whether the configuration is compatible or not. And if necessary, prepare the data!"""


if RANK == MASTER_RANK:
    _info_cache = {
        'start_time': -1.,
        'info_count': 0,
        'info_time': -1.,
    }
else:
    pass


def _clear_self():
    """Clear self to make sure the previous implementation does not mess things up!"""
    base['manifolds'] = dict()
    base['meshes'] = dict()
    base['spaces'] = dict()
    base['forms'] = dict()
    base['the_great_mesh'] = None

    if RANK == MASTER_RANK:
        _info_cache['start_time'] = -1.
        _info_cache['info_count'] = 0
        _info_cache['info_time'] = -1.
    else:
        pass


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
    # noinspection PyTypedDict,PyTypeChecker
    base['the_great_mesh'] = MseHttGreatMesh()


def _parse_spaces(abstract_spaces):
    r""""""
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
    r""""""
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
    r""""""
    the_great_mesh = base['the_great_mesh']
    partial_meshes = base['meshes']  # all the partial meshes
    spaces = base['spaces']          # all the msehtt spaces
    manifolds = base['manifolds']    # all the msehtt manifolds
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
    r"""The objects other than manifolds, meshes, spaces, root-forms that should be parsed for this
    particular fem setting.
    """
    if obj.__class__ is MatrixProxyLinearSystem:
        from phyem.msehtt.tools.linear_system.dynamic.main import MseHttDynamicLinearSystem
        dynamic = MseHttDynamicLinearSystem(obj, base)
        return dynamic
    elif obj.__class__ is MatrixProxyNoneLinearSystem:
        from phyem.msehtt.tools.nonlinear_system.dynamic.main import MseHttDynamicNonLinearSystem
        dynamic = MseHttDynamicNonLinearSystem(obj, base)
        return dynamic

    else:
        return None  # do not raise Error (like below)!
        # raise NotImplementedError(f"cannot parse msepy implementation for {obj}.")


class config:
    r""""""
    def __init__(self, obj):
        r""""""
        self._obj = obj

    def __call__(self, *args, **kwargs):
        r""""""
        _config(self._obj)(*args, **kwargs)


# noinspection PyUnresolvedReferences
def _config(obj=None):
    r""""""
    if obj is None or (isinstance(obj, str) and obj == 'tgm'):  # we are configuring the great mesh.
        obj = base['the_great_mesh']
    else:
        pass

    return obj._config


from phyem.msehtt.tools.gathering_matrix import ___clean_cache_msehtt_gm___
from phyem.msehtt.static.mesh.great.elements.types.base import ___clean_cache_msehtt_element_ct___


def clean_cache():
    r""""""
    ___clean_cache_msehtt_gm___()
    ___clean_cache_msehtt_element_ct___()


from time import time
from phyem.tools.miscellaneous.timer import MyTimer


___info_print_cache___ = {
    'last_count': 0,
    'last_time': -1.,
}


def info(*others_2b_printed):
    """We print the info, as much as possible, of the current msepy implementation."""
    # -- first we print the newest time of the cochain (if there is) of each form.
    if RANK != MASTER_RANK:
        return None
    else:
        pass

    if len(others_2b_printed) == 0:
        M = 0
        OTP = []
    else:
        m = others_2b_printed[0]

        if isinstance(m, int):
            M = m
            OTP = others_2b_printed[1:]
        else:
            M = 0
            OTP = others_2b_printed

    tbp = []

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
    tbp.append(f'==msehtt== [{count}] {MyTimer.current_time()} -after- %.2f(s), '
               f'total: {MyTimer.seconds2dhms(total_cost)} <----' % (new_time - old_time))
    tbp.append(f"~) Form with newest cochain @ --------- ")
    forms = base['forms']
    for form_sym in forms:
        form = forms[form_sym]
        if form._is_base():
            newest_time = form.cochain.newest
            if newest_time is not None:
                tbp.append('{:>20} @ {:<30}'.format(form.abstract._pure_lin_repr, newest_time))
        else:
            pass
    tbp.append(f"\n~) Existing time sequences --------- ")
    from phyem.src.time_sequence import _global_abstract_time_sequence
    for ats_lin in _global_abstract_time_sequence:
        ats = _global_abstract_time_sequence[ats_lin]
        _ = ats.info(return_str=True)
        if _ is None:
            pass
        else:
            tbp.append(_)

    tbp.append(f"\n~) Meshes:")
    meshes = base['meshes']
    for mesh_repr in meshes:
        mesh = meshes[mesh_repr]
        _ = mesh.info(return_str=True)
        if _ is None:
            pass
        else:
            tbp.append(_)

    tbp.append(f"\n~) Others: ~~~~")
    for i, other in enumerate(OTP):
        tbp.append(f"  {i}) -> {other}\n")

    tbp.append('\n\n')
    _info_cache['info_count'] = count + 1
    _info_cache['info_time'] = new_time

    if M == 0:  # info every single time
        tbp = '\n'.join(tbp)
        print(tbp, flush=True)
    else:
        time_gap = 60 * M
        if ___info_print_cache___['last_time'] == -1:
            ___info_print_cache___['last_time'] = old_time - time_gap - 1
        else:
            pass
        last_count = ___info_print_cache___['last_count']
        last_time = ___info_print_cache___['last_time']
        if (count - last_count) >= M or (new_time - last_time) > time_gap:
            tbp = '\n'.join(tbp)
            print(tbp, flush=True)
            ___info_print_cache___['last_count'] = count
            ___info_print_cache___['last_time'] = new_time
        else:
            pass
    return None


from phyem.src.spaces.main import _degree_str_maker


def array(indicator, *args, **kwargs):
    """Manually make an array."""
    if indicator == 'mass matrix':
        PARSER = base['PARSER']
        if (len(args) == 1) and (kwargs == {}):  # in this case, we must receive a msehtt static form.
            form = args[0]
            assert form.__class__ is MseHttForm, f"we must have received a msehtt static form."
            space = form.space.abstract._pure_lin_repr
            degree = _degree_str_maker(form.degree)
            # noinspection PyUnresolvedReferences
            M, time_caller = PARSER.Parse__M_matrix(space, degree, degree)
            return M, time_caller
        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError(f"CANNOT parse {indicator} array!")
