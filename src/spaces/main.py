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

_default_mass_matrix_reprs = [
    r"\mathsf{M}",
    _sep.join(["Mass:Mat", "{space_pure_lin_repr}", "{d0}", "{d1}"]),
]

_default_d_matrix_reprs = [
    r"\mathsf{D}",
    _sep.join(["d:Mat", "{space_pure_lin_repr}", "{d}"]),
]

_default_d_matrix_transpose_reprs = [
    r"\mathsf{D}",
    _sep.join(["d:T:Mat", "{space_pure_lin_repr}", "{d}"]),
]

_default_boundary_dp_vector_reprs = [
    # once we know f0, we can find the correct basis functions it wedged with
    r"\boldsymbol{b}",
    _sep.join(["BoundaryDP:Vec", "trStar[{f0}]", "tr[{f1}]"]),
    #                            <tr star bf0 | tr f1>.
]

_default_astA_x_astB_ip_tC_reprs = [
    r"\mathsf{c}",
    _sep.join(["c_ip:Mat", "[{A}]", "[{B}]", "[{C}]"]),
]

_default_astA_x_B_ip_tC_reprs = [
    r"\mathsf{C}",
    _sep.join(["X_ip:Mat", "[{A}]", "[{B}]", "[{C}]"]),
]

_default_A_x_astB_ip_tC_reprs = [
    r"\boldsymbol{C}",
    _sep.join(["C_ip:Mat", "[{A}]", "[{B}]", "[{C}]"]),
]

_default_A_x_B_ip_C_reprs = [
    r"\mathsf{X}",
    _sep.join(["X:Mat", "[{A}]", "[{B}]", "[{C}]"]),
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
