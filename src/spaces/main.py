# -*- coding: utf-8 -*-
r"""
"""
import sys

if './' not in sys.path:
    sys.path.append('./')
from importlib import import_module

_config = {
    'current_mesh': '',
}
_mesh_set = dict()
_space_set = dict()

_sep = ' ->- '

# whenever new space is implemented, add it below.
_implemented_spaces = {
    # indicator: (class path                    ,  class name            , description                 , parameters),
    'Lambda':    ('src.spaces.continuous.Lambda', 'ScalarValuedFormSpace', 'scalar valued k-form space', ['k', ]),
    'bundle':    ('src.spaces.continuous.bundle', 'BundleValuedFormSpace', 'bundle valued k-form space', ['k', ]),
    'bundle-diagonal': (
        'src.spaces.continuous.bundle_diagonal',
        'DiagonalBundleValuedFormSpace',
        'diagonal bundle valued k-form space',
        ['k', ]
    ),
}

__all__ = [
    '_VarSetting_mass_matrix',                     #
    '_VarSetting_d_matrix',                        #
    '_VarSetting_d_matrix_transpose',              #

    '_VarSetting_boundary_dp_vector',              #

    '_VarSetting_astA_x_astB_ip_tC',               #
    '_VarSetting_astA_x_B_ip_tC',                  #
    '_VarSetting_A_x_astB_ip_tC',                  #
    '_VarSetting_A_x_B_ip_C',                      # nonlinear

    '_VarSetting_dastA_astA_tp_tC',                #
    '_VarSetting_dastA_tB_tp_astA',
    '_VarSetting_dtA_astB_tp_astB',
    '_VarSetting_dA_B_tp_C__1Known',
    '_VarSetting_dA_B_tp_C__2Known',
    '_VarSetting_dA_B_tp_C',                       # nonlinear

    '_VarSetting_A_B_tp_C__1Known',
    '_VarSetting_A_B_tp_C__2Known',
    '_VarSetting_A_B_tp_C',                       # nonlinear

    '_VarSetting_IP_matrix_db_bf',                #
    '_VarSetting_IP_matrix_bf_db',                #
]

# ------ basic -----------------------------------------------------------------------------------
_VarSetting_mass_matrix = [
    r"\mathsf{M}",
    _sep.join(["Mass:Mat", "{space_pure_lin_repr}", "{d0}", "{d1}"]),
]

_VarSetting_d_matrix = [
    r"\mathsf{D}",
    _sep.join(["d:Mat", "{space_pure_lin_repr}", "{d}"]),
]

_VarSetting_d_matrix_transpose = [
    r"\mathbb{D}",
    _sep.join(["d:T:Mat", "{space_pure_lin_repr}", "{d}"]),
]

# Natural bc -------------------------------------------------------------------------------------

_VarSetting_boundary_dp_vector = [
    # once we know f0, we can find the correct basis functions it wedged with
    r"\boldsymbol{b}",
    _sep.join(["BoundaryDP:Vec", "trStar[{f0}]", "tr[{f1}]"]),
    #                            <tr star bf0 | tr f1>.
]

# (w x u, u) --------------------------------------------------------------------------------------

_VarSetting_astA_x_astB_ip_tC = [
    r"\mathsf{c}",
    _sep.join(["c_ip", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_astA_x_B_ip_tC = [
    r"\mathsf{C}",
    _sep.join(["X_ip", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_A_x_astB_ip_tC = [
    r"\boldsymbol{C}",
    _sep.join(["_Xip", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_A_x_B_ip_C = [
    r"\mathsf{X}",
    _sep.join(["_X_:", "[{A}]", "[{B}]", "[{C}]"]),
]

# -----(dA, B otimes C) --------------------------------------------------------------------------

_VarSetting_dastA_astA_tp_tC = [
    r"\left<\mathsf{d\cdot,\cdot\otimes\_}\right>",
    _sep.join(["d*A--*A-tp-tC", "[{A}]", "[{C}]"]),
]

_VarSetting_dastA_tB_tp_astA = [
    r"\left<\mathsf{d\cdot,\_\otimes\cdot}\right>",
    _sep.join(["d*A--tB-tp-*A", "[{A}]", "[{B}]"]),
]

_VarSetting_dtA_astB_tp_astB = [
    r"\left<\mathsf{d\_,\cdot\otimes\cdot}\right>",
    _sep.join(["dtA--*B-tp-*B", "[{A}]", "[{B}]"]),
]


_VarSetting_dastA_astA_tp_tC = [
    r"\left<\mathsf{d\cdot,\cdot\otimes\_}\right>",
    _sep.join(["d*A--*A-tp-tC", "[{A}]", "[{C}]"]),
]


_VarSetting_dA_B_tp_C__1Known = [   # A, B, C are different; and it must have a test form
    r"\left<\mathsf{d\_,\_\otimes\_}\right>",
    _sep.join(["dA--B-tp-C:1:Known", "[{A}]", "[{B}]", "[{C}]", "[{K}]", "[{T}]", "[{U}]"]),
]


_VarSetting_dA_B_tp_C__2Known = [  # A, B, C are different; two of them are known, and the rest one is the test form
    r"\left<\mathsf{d\_,\_\otimes\_}\right>",
    _sep.join(["dA--B-tp-C:2:Known", "[{A}]", "[{B}]", "[{C}]", "[{K1}]", "[{K2}]", "[{T}]"]),
]


_VarSetting_dA_B_tp_C = [  # A, B, C are different; # nonlinear
    r"\left<\mathsf{d\cdot,\cdot\otimes\cdot}\right>",
    _sep.join(["dA--B-tp-C", "[{A}]", "[{B}]", "[{C}]"]),
]


# (A, B otimes C) -------------------------------------------------------

_VarSetting_A_B_tp_C__1Known = [   # A, B, C are different; and it must have a test form
    r"\left<\mathsf{\_,\_\otimes\_}\right>",
    _sep.join(["A--B-tp-C:1:Known", "[{A}]", "[{B}]", "[{C}]", "[{K}]", "[{T}]", "[{U}]"]),
]


_VarSetting_A_B_tp_C__2Known = [  # A, B, C are different; two of them are known, and the rest one is the test form
    r"\left<\mathsf{\_,\_\otimes\_}\right>",
    _sep.join(["A--B-tp-C:2:Known", "[{A}]", "[{B}]", "[{C}]", "[{K1}]", "[{K2}]", "[{T}]"]),
]


_VarSetting_A_B_tp_C = [  # A, B, C are different; # nonlinear
    r"\left<\mathsf{\cdot,\cdot\otimes\cdot}\right>",
    _sep.join(["A--B-tp-C", "[{A}]", "[{B}]", "[{C}]"]),
]


# (bundle form, special diagonal bundle form)------------------------------------------------------

_VarSetting_IP_matrix_db_bf = [
    r"\mathbb{M}_{\mathcal{S}}",
    _sep.join(
        ["db-M-bf", "{db_space_pure_lin_repr}", "{bf_space_pure_lin_repr}", "{degree_db}", "{degree_bf}"]
    ),
]

_VarSetting_IP_matrix_bf_db = [
    r"\mathbb{M}^{\mathsf{T}}_{\mathcal{S}}",
    _sep.join(
        ["bf-M-db", "{bf_space_pure_lin_repr}", "{db_space_pure_lin_repr}", "{degree_bf}", "{degree_db}"]
    ),
]

_default_space_degree_repr = ':D-'

_degree_cache = {}


def _degree_str_maker(degree):
    """"""
    str_degree = degree.__class__.__name__ + str(degree)
    _degree_cache[str_degree] = degree
    return str_degree


def _str_degree_parser(str_degree):
    """"""
    return _degree_cache[str_degree]


def set_mesh(mesh):
    """"""
    assert mesh.__class__.__name__ == 'Mesh', \
        f"I need a Mesh instance."
    sr = mesh._sym_repr
    if sr in _mesh_set:
        pass
    else:
        _mesh_set[sr] = mesh
        _space_set[sr] = dict()

    _config['current_mesh'] = sr


def _list_spaces():
    """"""
    print('\n Implemented spaces:')
    print('{:>15} - {}'.format('abbreviation', 'description'))
    for abbr in _implemented_spaces:
        description = _implemented_spaces[abbr][2]
        print('{:>15} | {}'.format(abbr, description))

    print('\n Existing spaces:')
    for mesh in _space_set:
        spaces = _space_set[mesh]
        print('{:>15} {}'.format('On mesh', mesh))
        for i, sr in enumerate(spaces):
            space = spaces[sr]
            print('{:>15}: {}'.format(i, space._sym_repr))


def finite(degree, mesh=None, spaces=None):
    """

    Parameters
    ----------
    degree
    mesh
    spaces

    Returns
    -------

    """
    if mesh is None:  # do it for all spaces on all meshes.
        for mesh_sr in _mesh_set:
            mesh = _mesh_set[mesh_sr]
            finite(degree, mesh=mesh, spaces=spaces)
        return
    else:
        assert mesh.__class__.__name__ == 'Mesh', f"Mesh = {mesh} is not a Mesh object."
        mesh_sr = mesh._sym_repr

    all_current_spaces = _space_set[mesh_sr]

    if spaces is None:
        spaces = all_current_spaces.values()
    else:
        if not isinstance(spaces, (list, tuple)):
            spaces = [spaces, ]
        else:
            pass

    for sp in spaces:
        assert sp._sym_repr in all_current_spaces, f"space: {sp} is not a space in current mesh {mesh}."

    for space in spaces:
        space.finite.specify_all(degree)


def new(abbrs, *args, mesh=None, **kwargs):
    """generate a space (named `abbr`) with args `kwargs` use current mesh.

    Parameters
    ----------
    abbrs
    mesh :
        We specify a mesh here. And this does not change the global mesh setting.
    kwargs

    Returns
    -------

    """
    if _config['current_mesh'] == '' and mesh is None:
        raise Exception(f"pls set a mesh firstly by using `space.set_mesh` or specify `mesh`.")
    else:
        pass

    if isinstance(abbrs, str):  # make only 1 space
        abbrs = [abbrs, ]
    else:
        isinstance(abbrs, (list, tuple)), f"pls put space abbreviations into a list or tuple."

    mesh_sr = _config['current_mesh']

    if mesh is None:
        mesh = _mesh_set[mesh_sr]
    else:
        mesh_sr = mesh._sym_repr
        if mesh_sr in _mesh_set:
            pass
        else:
            _mesh_set[mesh_sr] = mesh
            _space_set[mesh_sr] = dict()

    current_spaces = _space_set[mesh_sr]

    spaces = tuple()

    for abbr in abbrs:

        assert abbr in _implemented_spaces, \
            f"space abbr.={abbr} not implemented. do `ph.space.list_()` to see all implemented spaces."

        space_class_path, space_class_name = _implemented_spaces[abbr][0:2]

        space_class = getattr(import_module(space_class_path), space_class_name)

        space = space_class(mesh, *args, **kwargs)

        srp = space._sym_repr  # do not use __repr__()

        if srp in current_spaces:
            pass
        else:
            current_spaces[srp] = space

        spaces += (current_spaces[srp],)

    if len(spaces) == 1:
        return spaces[0]

    else:
        return spaces
