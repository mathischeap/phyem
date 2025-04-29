# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
r"""

.. note::

    *phyem* is ready by itself with default global parameters. You can direct initialize a simulation with these
    default global parameters.

If you want to customize global parameters, you can do use following functions:

    .. autofunction:: set_embedding_space_dim

    .. autofunction:: set_high_accuracy

    .. autofunction:: set_pr_cache

For example,

>>> ph.config.set_embedding_space_dim(2)
>>> ph.config.set_high_accuracy(True)
>>> ph.config.set_pr_cache(False)

These commands set the embedding space to be 2-dimensional, ask *phyem* to have a high accuracy and not
to use *pr* cache (or to present *pr* outputs in real time). These parameters are same to the default values; they have
no effects (thus we preferably omit them).

"""

_global_variables = {
    'embedding_space_dim': 2,      # default embedding_space_dim is 2.
    'zero_entry_threshold': 1e-12  # set to be zero when it is lower than this
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


_setting = {
    "block": True,   # matplot block
    "cache_folder": '__phcache__',  # the default cache folder name
    "pr_cache": False,   # if this is True, all pr will save to the cache folder and will not show()
    "pr_cache_maximum": 3,  # We clean toward this number when it is more than twice
    "pr_cache_counter": 0,  # we have cache how many times?
    "pr_cache_subfolder": '',  # we cache where in particular?
    "pr_cache_folder_prefix": 'Pr_',  # we cache where in particular?
    "pr_cache_current_folder": 'current',  # we cache where in particular?
    "high_accuracy": True,  # influence sparsity of some matrices.
    "auto_cleaning": True,  # automatic memory saving
}


def set_embedding_space_dim(ndim):
    """To set the dimensions of the space where our simulation problem is defined, i.e., the embedding space.
    The default dimensions are 2.

    Parameters
    ----------
    ndim : {1, 2, 3}
        *phyem* can only simulate problems in 1-, 2- or 3-d space.

    """
    assert ndim % 1 == 0 and ndim > 0, f"ndim={ndim} is wrong, it must be a positive integer."
    _global_variables['embedding_space_dim'] = ndim
    _clear_all()   # whenever we change or reset the space dim, we clear all abstract objects.
    _clear_pr_cache()


def set_high_accuracy(_bool):
    """*phyem* may trade for higher accuracy in the cost of losing a little performance. Turn this feature
    on or off through this function. The default value is ``True``.

    For example, *phyem* can use numerical quadrature of a higher degree (thus of higher accuracy)
    to compute the mass matrices, which could
    decrease the sparsity and sequentially slow down the assembling and solving processes slightly.

    Parameters
    ----------
    _bool : bool
        If ``_bool`` is ``True``, *phyem* occasionally has a higher accuracy.
        The real effect depends on the problem you are solving.
    """
    assert isinstance(_bool, bool), f"give me True or False"
    _setting['high_accuracy'] = _bool


def set_pr_cache(_bool):
    """ The abbreviation `pr` stands for `print representation`.
    Many classes in *phyem* have the ``pr`` method. Calling it of an instance by ``.pr()`` will usually invoke
    the ``matplotlib`` and ``LaTeX`` pakages to render a picture which proper illustrates the instance.
    By turning the *pr* cache on, all outputs resulting from ``pr`` methods will be saved to
    ``./__phcache__/Pr_current/`` instead of being shown in real time.

    The default value is ``False``.

    Parameters
    ----------
    _bool : bool
        If ``_bool`` is ``True``, all outputs resulting from ``pr`` methods will be saved
        to ``./__phcache__/Pr_current/``.

    """
    assert isinstance(_bool, bool), f"give me True or False"
    _setting['pr_cache'] = _bool


def set_auto_cleaning(_bool_or_factor):
    """

    Parameters
    ----------
    _bool_or_factor

    Returns
    -------

    """
    if isinstance(_bool_or_factor, bool):
        pass
    elif isinstance(_bool_or_factor, (int, float)):
        _bool_or_factor = int(_bool_or_factor)
        assert _bool_or_factor >= 2, f"must set clean factor >= 2."
    else:
        raise Exception(f"clean factor to be a bool or a positive integer.")
    _setting['auto_cleaning'] = _bool_or_factor


def _pr_cache(fig, filename=None):
    """

    Parameters
    ----------
    fig
    filename :
        The filename received from particular pr method. It will be put into the final filename.

    Returns
    -------

    """
    if RANK != MASTER_RANK:
        return
    else:
        pass

    from tools.os_ import mkdir, empty_dir, isdir
    phcache_folder = _setting[r'cache_folder']
    if isdir(phcache_folder):
        pass
    else:
        mkdir(phcache_folder)

    from time import time
    from tools.miscellaneous.random_ import string_digits
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')

    if filename is None:
        filename_personal = rf'/{_setting["pr_cache_counter"]}.png'
    else:
        filename_personal = rf'/{_setting["pr_cache_counter"]}_{filename}.png'

    if _setting["pr_cache_counter"] == 0:
        str_time = str(time()).split('.')[0]
        subfolder_name = (
            phcache_folder + r"/" + _setting["pr_cache_folder_prefix"] +
            str_time + '_' + string_digits(12)  # how we name the random folder
        )
        _setting["pr_cache_subfolder"] = subfolder_name
        mkdir(subfolder_name)
    else:
        subfolder_name = _setting["pr_cache_subfolder"]
        assert subfolder_name != '', f"something is wrong!"

    plt.savefig(subfolder_name + filename_personal, bbox_inches='tight', dpi=200)

    folder_current = (
        phcache_folder + '/' + _setting["pr_cache_folder_prefix"] + _setting["pr_cache_current_folder"]
    )

    if _setting["pr_cache_counter"] == 0:
        mkdir(folder_current)   # do nothing if the folder exists
        empty_dir(folder_current)  # remove old files if there is any.
    else:
        pass

    plt.savefig(folder_current + filename_personal, bbox_inches='tight', dpi=200)

    _setting["pr_cache_counter"] += 1
    plt.close(fig)


def _set_matplot_block(block):
    r""""""
    _setting['block'] = block


def _clear_pr_cache():
    r""""""
    _setting["pr_cache_counter"] = 0  # reset cache counting
    _setting["pr_cache_subfolder"] = ''  # clean cache_subfolder

    if RANK == MASTER_RANK:
        from tools.os_ import listdir, isdir, mkdir, rmdir, empty_dir

        phcache_folder = _setting[r'cache_folder']
        if isdir(phcache_folder):
            pass
        else:
            mkdir(phcache_folder)
        all_ph_cache_files = listdir(phcache_folder)  # including folder names.

        pr_prefix = _setting['pr_cache_folder_prefix']
        current_file = _setting["pr_cache_folder_prefix"] + _setting["pr_cache_current_folder"]
        len_prefix = len(pr_prefix)
        number_pr_files = 0  # excluding the current file
        pr_files = list()  # excluding the current file
        for cache_file in all_ph_cache_files:
            if cache_file[:len_prefix] == pr_prefix and cache_file != current_file:
                number_pr_files += 1
                pr_files.append(cache_file)
            else:
                pass
        pr_cache_maximum = _setting["pr_cache_maximum"]
        assert pr_cache_maximum > 1 and pr_cache_maximum % 1 == 0, \
            f"_setting['pr_cache_maximum'] = {pr_cache_maximum} is illegal, give me a positive integer (>1)."
        if number_pr_files >= 2 * pr_cache_maximum:
            files_2b_clean = pr_files[:pr_cache_maximum + 1]
            for file in files_2b_clean:
                empty_dir(phcache_folder + r"/" + file)
                rmdir(phcache_folder + r"/" + file)
        else:
            pass
    else:
        pass


def _clear_all():
    r"""clear all abstract objects.

    Make sure that, when we add new global cache, put it here.
    """
    from src.algebra.array import _global_root_arrays
    from src.algebra.nonlinear_operator import _global_nop_arrays
    from src.form.main import _global_forms
    from src.form.main import _global_root_forms_lin_dict
    _clear_a_dict(_global_root_arrays)
    _clear_a_dict(_global_nop_arrays)
    _clear_a_dict(_global_forms)
    _clear_a_dict(_global_root_forms_lin_dict)
    from src.form.parameters import _global_root_constant_scalars
    _clear_a_dict(_global_root_constant_scalars)
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

    from src.wf.term.main import _global_wf_terms
    _clear_a_dict(_global_wf_terms)


def _clear_a_dict(the_dict):
    """"""
    keys = list(the_dict.keys())
    for key in keys:
        del the_dict[key]


def get_embedding_space_dim():
    """"""
    return _global_variables['embedding_space_dim']


# lib setting config ...............
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
    'multidimensional_array': [r'\textlf{', r'}'],
    'TermNonLinearAlgebraicProxy': [r'\textmd{', r'}'],
}


def _parse_lin_repr(obj, pure_lin_repr):
    r""""""
    assert isinstance(pure_lin_repr, str) and len(pure_lin_repr) > 0, \
        f"linguistic_representation must be str of length > 0."
    assert all([_ not in r"{$\}" for _ in pure_lin_repr]), (
            f"pure_lin_repr={pure_lin_repr} illegal, cannot contain" + r"'{\}'.")
    start, end = _global_lin_repr_setting[obj]
    return start + pure_lin_repr + end, pure_lin_repr


def _parse_type_and_pure_lin_repr(lin_repr):
    r""""""
    for what in _global_lin_repr_setting:
        key = _global_lin_repr_setting[what][0]
        lk = len(key)
        if lin_repr[:lk] == key:
            return what, lin_repr[lk:-1]


_manifold_default_sym_repr = r'\mathcal{M}'
_mesh_default_sym_repr = r'\mathfrak{M}'
_abstract_time_sequence_default_sym_repr = r'\mathtt{T}^S'
_abstract_time_interval_default_sym_repr = r'\Delta t'


_mesh_partition_sym_repr = [r"\partial\mathfrak{M}\left(", r"\right)"]
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
    'convect': r"$\cdot\nabla$",
    'tensor_product': r"$\otimes$",

    'projection': r'$\pi$ '
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

    'cross_product': r"{\times}",
    'tensor_product': r"{\otimes}",
    'convect': r"{\cdot\nabla}",

    'projection': r'{\pi}'
}


_wf_term_default_simple_patterns = {   # use only str to represent a simple pattern.
    # indicator : simple pattern
    '(pt,)': '(partial_t rf, f)',   # (partial_time_derivative of root-sf, sf)
    '(cd,)': '(codifferential f, f)',

    # below, we have simple patterns only for root-sf.
    '(rt,rt)': '(rf, rf)',
    '<|>': r"$\left<\left. \mathrm{rf} \right| \mathrm{rf} \right>$",              # <A|B>
    '(d,)': '(d rf, rf)',
    '<d,>': '<d rf, dual-rf>',
    '(,d)': '(rf, d rf)',
    '<,d>': '<dual-rf, d rf>',

    '(d,d)': '(d rf, d rf)',

    '(d(*A),B)': r"$\left(\mathrm{d}\left(\star A\right),B\right)$",

    '(<db>,d<b>)': '(root-diagonal-bf, d root-bf)',

    '<tr star | tr >': r'$\left\langle\right.$ tr star rf $|$ trace rf$\left.\right\rangle$',
    '<tr star | tr > - restrict': r'$\left\langle\right.$ tr star rf $||$ trace rf$\left.\right\rangle$',

    '(A, trB)': r"$(A, \mathrm{tr} B)$",
    '(trB, A)': r"$(\mathrm{tr} B, A)$",

    '<*x*|C>': r'$\left<\left.\mathrm{krf} \times \mathrm{krf} \right| \mathrm{rf}\right>$',
    '<Ax*|C>': r'$\left<\left.\mathrm{rf} \times \mathrm{krf} \right| \mathrm{rf}\right>$',
    '<*xB|C>': r'$\left<\left.\mathrm{krf} \times \mathrm{rf} \right| \mathrm{rf}\right>$',
    '<AxB|C>': r'$\left<\left.\mathrm{rf} \times \mathrm{rf} \right| \mathrm{rf}\right>$',

    '<*x*|d(C)>': r'$\left<\left.\mathrm{krf} \times \mathrm{krf} \right| \mathrm{d}(\mathrm{rf})\right>$',

    '(* .V *, C)': r"$\left(\mathrm{krf} \cdot\nabla \mathrm{krf}, \mathrm{rf}\right)$",

    '<AxB|CxD>': r"$\left<\left.\mathrm{rf} \times \mathrm{rf} \right| \mathrm{rf} \times \mathrm{rf}\right>$",
    '<*x*|*xD>': r"$\left<\left.\mathrm{krf} \times \mathrm{krf} \right| \mathrm{krf} \times \mathrm{rf}\right>$",

    '(,d-pi)': '(rf, d(pi(rf)))',

    '(*x*,)': r'(krf $\times$ krf, rf)',   # vector
    '(*x,)': r'(krf $\times$ rf, rf)',
    '(x*,)': r'(rf $\times$ krf, rf)',

    '(*x*,*x)': r"(krf $\times$ krf, krf $\times$ rf)",  # vector , (*A x *B, *C x D)
    '(x*,*x)': r"(rf $\times$ krf, krf $\times$ rf)",    # matrix , (A x *B, *C x D)

    '(*x,d)': r"(krf $\times$ rf, d rf)",
    '(x*,d)': r"(rf $\times$ krf, d rf)",
    '(*x*,d)': r"(krf $\times$ krf, d rf)",   # vector
    '(AxB,dC)': r"(rf $\times$ rf, d rf)",

    '(x,)': r'(rf $\times$ rf, rf)',   # nonlinear term.

    '(d0*,0*tp)': r'(d krf-0, krf-0 $\otimes$ rf)',
    '(d0*,tp0*)': r'(d krf-0, rf $\otimes$ krf-0)',
    '(d,0*tp0*)': r'(d rf, krf-0 $\otimes$ krf-0)',
    '(d,tp):1K': r'(d rf, rf $\otimes$ rf):1K',
    '(d,tp):2K': r'(d rf, rf $\otimes$ rf):2K',
    '(d,tp)': r'(d rf, rf $\otimes$ rf)',

    '(,tp):1K': r'(rf, rf $\otimes$ rf):1K',
    '(,tp):2K': r'(rf, rf $\otimes$ rf):2K',
    '(,tp)': r'(rf, rf $\otimes$ rf)',
}

_pde_test_form_lin_repr = 'th-test-form'

_nonlinear_ap_test_form_repr = {
    'sym': r' \wr ',
    'lin': ' >=~=< ',
}
