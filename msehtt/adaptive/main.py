# -*- coding: utf-8 -*-
r"""
"""
base = {
    'manifolds': dict(),  # keys: abstract manifold sym_repr
    'meshes': None,   # all forms/meshes will also point to this great mesh.
    'spaces': None,
    'forms': None,
    'the_great_mesh': None,

    # ---------- EXECUTION RELATED ----------------------------------------------------------------------
    '___config_meshes___': [],  # to make sure meshes are initialized in the sequence as they configured.
    '_BASE_': None,
    "current_generation": -1,
    "current_base": None,
    'stamp': '',
}


from time import time
from tools.miscellaneous.random_ import string_digits
from tools.miscellaneous.timer import MyTimer

from msehtt.static.mesh.great.main import MseHttGreatMesh

from msehtt.static.manifold.main import MseHttManifold

from msehtt.adaptive.mesh.main import MseHtt_Adaptive_TopMesh
from msehtt.adaptive.space.main import MseHtt_Adaptive_TopSpace
from msehtt.adaptive.form.main import MseHtt_Adaptive_TopForm

from src.wf.mp.linear_system import MatrixProxyLinearSystem
from src.wf.mp.nonlinear_system import MatrixProxyNoneLinearSystem

from msehtt.adaptive.tools.linear_system import MseHtt_Adaptive_Linear_System
from msehtt.adaptive.tools.nonlinear_system import MseHtt_Adaptive_NonLinear_System

from src.config import RANK, MASTER_RANK, COMM, SIZE


from msehtt.tools.gathering_matrix import ___clean_cache_msehtt_gm___
from msehtt.static.mesh.great.elements.types.base import ___clean_cache_msehtt_element_ct___


def clean_cache():
    r""""""
    ___clean_cache_msehtt_gm___()
    ___clean_cache_msehtt_element_ct___()


___MAX_GENERATIONS____ = 2


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
    base['meshes'] = None
    base['spaces'] = None
    base['forms'] = None
    base['the_great_mesh'] = None
    base['___config_meshes___'] = []
    base['_BASE_'] = None
    base['current_generation'] = -1
    base['current_base'] = None
    base['stamp'] = ''

    if RANK == MASTER_RANK:
        _info_cache['start_time'] = -1
        _info_cache['info_count'] = 0
        _info_cache['info_time'] = -1
    else:
        pass


def tgm():
    """return the great mesh."""
    return base['the_great_mesh']


def _parse_manifolds(abstract_manifolds):
    """"""
    manifold_dict = {}
    for sym in abstract_manifolds:
        manifold = MseHttManifold(abstract_manifolds[sym])
        manifold_dict[sym] = manifold
    base['manifolds'] = manifold_dict


def _parse_meshes(abstract_meshes):
    """"""
    mesh_dict = {}
    for sym in abstract_meshes:
        am = abstract_meshes[sym]
        pm = MseHtt_Adaptive_TopMesh(am, ___MAX_GENERATIONS____)
        mesh_dict[sym] = pm
    assert base['meshes'] is None, f"We must do not generate the top mesh yet."
    base['meshes'] = mesh_dict
    # noinspection PyTypeChecker
    base['the_great_mesh'] = MseHttGreatMesh()


def _parse_spaces(abstract_spaces):
    r""""""
    space_dict = {}
    for ab_msh_sym_repr in abstract_spaces:
        ab_sps = abstract_spaces[ab_msh_sym_repr]

        for ab_sp_sym_repr in ab_sps:
            ab_sp = ab_sps[ab_sp_sym_repr]
            space = MseHtt_Adaptive_TopSpace(ab_sp, ___MAX_GENERATIONS____)
            space_dict[ab_sp_sym_repr] = space

    base['spaces'] = space_dict


def _parse_root_forms(abstract_rfs):
    r""""""
    rf_dict = {}
    for rf_lin_repr in abstract_rfs:  # do it for all general root-forms
        abs_rf = abstract_rfs[rf_lin_repr]
        pure_lin_repr = abs_rf._pure_lin_repr
        rf_dict[pure_lin_repr] = MseHtt_Adaptive_TopForm(abs_rf, ___MAX_GENERATIONS____)
    base['forms'] = rf_dict


def _parse(obj):
    r"""The objects other than manifolds, meshes, spaces, root-forms that should be parsed for this
    particular fem setting.
    """
    if obj.__class__ is MatrixProxyLinearSystem:
        adapt_dynamic_LS = MseHtt_Adaptive_Linear_System(obj)
        adapt_dynamic_LS._msehtt_adaptive_ = current_base
        return adapt_dynamic_LS
    elif isinstance(obj, MatrixProxyNoneLinearSystem):
        adapt_dynamic_nLS = MseHtt_Adaptive_NonLinear_System(obj)
        adapt_dynamic_nLS._msehtt_adaptive_ = current_base
        return adapt_dynamic_nLS
    else:
        return None


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

    if isinstance(obj, MseHtt_Adaptive_TopMesh):
        ___config_meshes___ = base['___config_meshes___']
        assert obj.abstract._sym_repr not in ___config_meshes___, \
            f"mesh {obj.abstract._sym_repr} is already configured."
        ___config_meshes___.append(obj.abstract._sym_repr)
        # to make sure meshes are initialized in the sequence as they configured.
    else:
        pass

    return obj._config


from msehtt.adaptive.___func___ import ___base_tgm___
from msehtt.adaptive.___func___ import ___link_all_forms____
from msehtt.adaptive.___func___ import ___func_renew___
from msehtt.adaptive.___func___ import ___renew_stamp___
from msehtt.adaptive.___func___ import ___check_tgm___


def initialize(ts=1):
    r"""We can and should initialize the setting after all meshes are configured."""
    ___check_tgm___(base['the_great_mesh'])
    ___renew___(0, ts=ts)


def renew(trf, ts=1, use_method=3, clean=False, forms=None):
    r""""""
    ___renew___(trf=trf, ts=ts)

    form_renew_info = ___renew_root_form_cochains___(use_method=use_method, clean=clean, forms=forms)

    COMM.barrier()

    # --------- merge renew info --------------------------------------------------------
    overall_info = {}

    ALL_SAME = True
    for sym in form_renew_info:
        one_form_info = form_renew_info[sym]
        if 'all same' in one_form_info:
            all_same = one_form_info['all same']
            if all_same:
                pass
            else:
                ALL_SAME = False
        else:
            pass
    overall_info['all same'] = ALL_SAME
    overall_info['forms'] = form_renew_info

    return overall_info


def ___renew___(trf, ts=1):
    r""""""

    ___clean_cache_msehtt_gm___()

    COMM.barrier()

    _base_ = ___base_tgm___(base)
    new_tgm = MseHttGreatMesh()
    new_tgm._config(_base_, trf=trf, ts=ts)

    ___func_renew___(new_tgm, base)

    # ---------- link all forms ---------------------------------
    ___link_all_forms____(new_tgm, base)

    if RANK == MASTER_RANK:
        renew_stamp = string_digits(32) + '@' + str(time())
    else:
        renew_stamp = None
    renew_stamp = COMM.bcast(renew_stamp, root=MASTER_RANK)

    ___renew_stamp___(renew_stamp, trf, ts, base)


def ___renew_root_form_cochains___(use_method=3, clean=False, from_generation=-2, to_generation=-1, forms=None):
    r""""""
    form_renew_info = {}
    for pur_lin_repr in base['forms']:
        form = base['forms'][pur_lin_repr]
        do_for_this_form = True
        if forms is None:  # do it for all forms.
            pass
        else:
            if isinstance(forms, MseHtt_Adaptive_TopForm):
                if form is forms:
                    pass
                else:
                    do_for_this_form = False
            elif isinstance(forms, (list, tuple)):
                if form in forms:
                    pass
                else:
                    do_for_this_form = False
            else:
                raise NotImplementedError()

        if do_for_this_form:
            num_cochain = len(form._generations_[from_generation].cochain)
            if num_cochain == 0:
                pass
            else:
                is_base = form._generations_[from_generation]._is_base()
                if is_base:
                    one_form_renew_info = form.___renew_cochains___(
                        from_generation=from_generation,
                        to_generation=to_generation,
                        use_method=use_method,
                        clean=clean
                    )
                    form_renew_info[pur_lin_repr] = one_form_renew_info
                else:
                    pass
        else:
            pass

    return form_renew_info


import msehtt.static.implementation_array_parser as PARSER
import msehtt.static.implementation_nop_parser as NOC_PARSER


def current_base():
    r""""""
    ___config_meshes___ = base['___config_meshes___']
    TGM = None
    generations = None

    # check_current_generation --------------------------------
    for key in ___config_meshes___:
        mesh = base['meshes'][key]
        if TGM is None:
            TGM = mesh.current.tgm
            generations = mesh.ith_generation
        else:
            assert mesh.current.tgm is TGM
            assert generations == mesh.ith_generation

    for key in base['spaces']:
        space = base['spaces'][key]
        assert space.current.tgm is TGM
        assert generations == space.ith_generation

    for key in base['forms']:
        form = base['forms'][key]
        assert form.current.tgm is TGM
        assert generations == form.ith_generation

    if generations == base['current_generation']:
        return generations, base['current_base']
    else:
        pass
    # ============================================================

    assert TGM is not None, f'Must have a current great mesh.'

    C_BASE = {}
    C_FORMS = {}
    C_SPACES = {}
    C_MESHES = {}
    for key in ___config_meshes___:
        C_MESHES[key] = base['meshes'][key].current
    for key in base['spaces']:
        C_SPACES[key] = base['spaces'][key].current
    for key in base['forms']:
        C_FORMS[key] = base['forms'][key].current
    C_BASE['meshes'] = C_MESHES
    C_BASE['spaces'] = C_SPACES
    C_BASE['forms'] = C_FORMS
    C_BASE['the_great_mesh'] = TGM

    PARSER._setting_['base'] = C_BASE
    NOC_PARSER._setting_['base'] = C_BASE

    PARSER._setting_['_cache_M_'] = {}  # clean cache
    PARSER.AxB_ip_C.clean_cache()
    PARSER.AxB_dp_C.clean_cache()
    NOC_PARSER.AxB_ip_dC.clean_cache()

    C_BASE['PARSER'] = PARSER
    C_BASE['NOC-PARSER'] = NOC_PARSER

    base['current_generation'] = generations
    base['current_base'] = C_BASE

    return generations, C_BASE


def info(*others_2b_printed):
    """We print the info, as much as possible, of the current msepy implementation."""
    # -- first we print the newest time of the cochain (if there is) of each form.
    if RANK != MASTER_RANK:
        return None
    else:
        pass

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
    print(f'= msehtt-A [{count}] {MyTimer.current_time()} -after- %.2f(s)'
          f', total: {MyTimer.seconds2dhms(total_cost)} -n<{SIZE}> =' % (new_time - old_time))
    print(f"~) Form with newest cochain @ --------- ")
    forms = base['forms']
    for form_sym in forms:
        top_form = forms[form_sym]
        if len(top_form._generations_) > 0:
            form = top_form.current
            if form._is_base():
                newest_time = form.cochain.newest
                if newest_time is not None:
                    print(
                        '{:>20} @ {:<24}'.format(form.abstract._pure_lin_repr, newest_time) +
                        f'---> G[{top_form.ith_generation}]'
                    )
            else:
                pass
        else:
            pass
    print(f"\n~) Existing time sequences --------- ")
    from src.time_sequence import _global_abstract_time_sequence
    for ats_lin in _global_abstract_time_sequence:
        ats = _global_abstract_time_sequence[ats_lin]
        ats.info()

    print(f"\n~) Meshes:")
    meshes = base['meshes']
    for mesh_repr in meshes:
        top_mesh = meshes[mesh_repr]
        if len(top_mesh._generations_) == 0:
            pass
        else:
            mesh = top_mesh.current
            mesh.info(f" G[{top_mesh.ith_generation}] ")

    print(f"\n~) Others: ~~~~")
    for i, other in enumerate(others_2b_printed):
        print(f"  {i}) -> {other}\n")

    print('\n\n', flush=True)
    _info_cache['info_count'] = count + 1
    _info_cache['info_time'] = new_time
    return None
