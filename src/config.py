# -*- coding: utf-8 -*-
r"""Global configuring variables and methods.

@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
@time: 11/26/2022 2:56 PM
"""

_global_variables = {
    'embedding_space_dim': 3,  # default embedding_space_dim is 3.
}


# MPI setting
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK: int = COMM.Get_rank()
    SIZE: int = COMM.Get_size()
except ModuleNotFoundError:  # no `mpi4py` installed.
    COMM = None
    RANK = 0
    SIZE: int = 1

MASTER_RANK: int = 0  # DO NOT change this

_matplot_setting = {
    'block': True
}


def _set_matplot_block(block):
    _matplot_setting['block'] = block


# space config
def set_embedding_space_dim(ndim):
    """"""
    assert ndim % 1 == 0 and ndim > 0, f"ndim={ndim} is wrong, it must be a positive integer."
    _global_variables['embedding_space_dim'] = ndim
    _clear_all()   # whenever we change the space dim, we clear all abstract objects.


def _clear_all():
    """clear all abstract objects."""
    from src.algebra.array import _global_root_arrays
    from src.form.main import _global_forms
    from src.form.main import _global_root_forms_lin_dict
    _clear_a_dict(_global_root_arrays)
    _clear_a_dict(_global_forms)
    _clear_a_dict(_global_root_forms_lin_dict)
    from src.form.main import _global_form_variables
    _global_form_variables['update_cache'] = True

    from src.manifold import _global_manifolds
    _clear_a_dict(_global_manifolds)
    from src.mesh import _global_meshes
    _clear_a_dict(_global_meshes)

    from src.time_sequence import _global_abstract_time_sequence
    from src.time_sequence import _global_abstract_time_interval
    _clear_a_dict(_global_abstract_time_sequence)
    _clear_a_dict(_global_abstract_time_interval)

    from src.spaces.main import _config
    _config['current_mesh'] = ''
    from src.spaces.main import _degree_cache
    from src.spaces.main import _space_set
    from src.spaces.main import _mesh_set
    _clear_a_dict(_degree_cache)
    _clear_a_dict(_space_set)
    _clear_a_dict(_mesh_set)


def _clear_a_dict(the_dict):
    """"""
    keys = list(the_dict.keys())
    for key in keys:
        del the_dict[key]


def get_embedding_space_dim():
    """"""
    return _global_variables['embedding_space_dim']


# lib setting config
_abstract_time_sequence_default_lin_repr = 'Ts'
_manifold_default_lin_repr = 'Manifold'
_mesh_default_lin_repr = 'Mesh'


_global_lin_repr_setting = {
    # objects
    'manifold': [r'\underline{', '}'],
    'mesh': [r'\textrm{', r'}'],
    'form': [r'\textsf{', r'}'],
    'scalar_parameter': [r'\textsc{', r'}'],
    'abstract_time_sequence': [r'\textit{', r'}'],
    'abstract_time_interval': [r'\texttt{', r'}'],   # do not use `textsc`.
    'abstract_time_instant': [r'\textsl{', r'}'],
    'array': [r'\textbf{', r'}'],
}


def _parse_lin_repr(obj, lin_repr):
    """"""
    assert isinstance(lin_repr, str) and len(lin_repr) > 0, f"linguistic_representation must be str of length > 0."
    assert all([_ not in r"{$\}" for _ in lin_repr]), f"lin_repr={lin_repr} illegal, cannot contain" + r"'{\}'."
    start, end = _global_lin_repr_setting[obj]
    return start + lin_repr + end, lin_repr


def _parse_type_and_pure_lin_repr(lin_repr):
    """"""
    for what in _global_lin_repr_setting:
        key = _global_lin_repr_setting[what][0]
        lk = len(key)
        if lin_repr[:lk] == key:
            return what, lin_repr[lk:-1]


_manifold_default_sym_repr = r'\mathcal{M}'
_mesh_default_sym_repr = r'\mathfrak{M}'
_abstract_time_sequence_default_sym_repr = r'\mathtt{T}^S'
_abstract_time_interval_default_sym_repr = r'\Delta t'


_mesh_partition_sym_repr = [r"M_{sh}\left(", r"\right)"]
_mesh_partition_lin_repr = r"mesh-over: "

_manifold_partition_lin_repr = "=sub"


def _check_sym_repr(sym_repr):   # not used for forms as they have their own checker.
    """"""
    assert isinstance(sym_repr, str), f"sym_repr = {sym_repr} illegal, must be a string."
    pure_sym_repr = sym_repr.replace(' ', '')
    assert len(pure_sym_repr) > 0, f"sym_repr={sym_repr} illegal, it cannot be empty."
    return sym_repr


_form_evaluate_at_repr_setting = {
    'sym': [r"\left.", r"\right|^{(", ")}"],
    'lin': "@",
}

_root_form_ap_vec_setting = {
    'sym': [r"\vec{", r"}"],
    'lin': "+vec"
}

_transpose_text = '-transpose'

_abstract_array_factor_sep = r'\{*\}'
_abstract_array_connector = r'\{@\}'


_global_operator_lin_repr_setting = {  # coded operators
    'plus': r" $+$ ",
    'minus': r" $-$ ",
    'wedge': r" $\wedge$ ",
    'Hodge': r'$\star$ ',
    'd': r'$\mathrm{d}$ ',
    'codifferential': r'$\mathrm{d}^{\ast}$ ',
    'time_derivative': r'$\partial_{t}$ ',
    'trace': r'\emph{tr} ',

    'L2-inner-product': [r"$($", r'\emph{,} ', r"$)$ \emph{over} "],
    'duality-pairing': [r"$<$", r'\emph{,} ', r"$>$ \emph{over} "],

    'division': r' \emph{divided by} ',
    'multiply': r' \emph{multiply} ',

    'cross_product': r"$\times$",
}

_non_root_lin_sep = [r'\{', r'\}']


_global_operator_sym_repr_setting = {  # coded operators
    'plus': r"+",
    'minus': r"-",
    'wedge': r"{\wedge}",
    'Hodge': r'{\star}',
    'd': r'\mathrm{d}',
    'codifferential': r'\mathrm{d}^{\ast}',
    'time_derivative': r'\partial_{t}',
    'trace': r'\mathrm{tr}',
    'division': [r'\frac{', r'}{', r"}"],
    'cross_product': r"{\times}"
}


_wf_term_default_simple_patterns = {   # use only str to represent a simple pattern.
    # indicator : simple pattern
    '(pt,)': '(partial_t root-sf, sf)',   # (partial_time_derivative of root-sf, sf)
    '(cd,)': '(codifferential sf, sf)',

    # below, we have simple patterns only for root-sf.
    '(rt,rt)': '(root-sf, root-sf)',
    '(d,)': '(d root-sf, root-sf)',
    '(,d)': '(root-sf, d root-sf)',
    '<tr star | tr >': '<tr star root-sf | trace root-sf>',

    '(*x,)': '(known-root-sf cross-product root-sf, root-sf)',
}
