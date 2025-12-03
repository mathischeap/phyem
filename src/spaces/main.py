# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
r"""

.. _docs-space:

Space
=====

Having abstract meshes, you can define abstract finite dimensional function spaces on them.
To do so, we first need to set the target mesh by calling

>>> ph.space.set_mesh(mesh)

Note that if there is only one mesh defined, above command can be omitted.

Then, to define a finite dimensional function space on this mesh, call function ``ph.space.new``,

    .. autofunction:: phyem.src.spaces.main.new

So far, we have implemented the following spaces.

.. admonition:: Implemented spaces

    +-------------------------+-----------------+------------------------------+-------------------------------------+
    | **description**         |**abbr.**        |    **arg**                   |    **kwarg**                        |
    +-------------------------+-----------------+------------------------------+-------------------------------------+
    | scalar-valued form      | ``'Lambda'``    | ``k`` : int.                 |   ``orientation``:                  |
    | space                   |                 | It is a :math:`k`-form       |   {``'inner'``, ``'outer'``}.       |
    |                         |                 | space.                       |   The orientation of the form space.|
    |                         |                 |                              |   The default orientation is        |
    |                         |                 |                              |   ``'outer'``.                      |
    |                         |                 |                              |                                     |
    +-------------------------+-----------------+------------------------------+-------------------------------------+

For example, to make spaces of outer orientated 1-forms and 2-forms, do

>>> Out1 = ph.space.new('Lambda', 1, orientation='outer')
>>> Out2 = ph.space.new('Lambda', 2, orientation='outer')

And we can list all existing spaces by calling ``ph.list_spaces`` method,

>>> ph.list_spaces()  # doctest: +ELLIPSIS
Implemented spaces:...

.. automodule:: phyem.src.spaces.base
    :undoc-members:

"""


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
    'Lambda':    ('phyem.src.spaces.continuous.Lambda', 'ScalarValuedFormSpace', 'scalar valued k-form space', ['k', ]),
    'bundle':    ('phyem.src.spaces.continuous.bundle', 'BundleValuedFormSpace', 'bundle valued k-form space', ['k', ]),
    'bundle-diagonal': (
        'phyem.src.spaces.continuous.bundle_diagonal',
        'DiagonalBundleValuedFormSpace',
        'diagonal bundle valued k-form space',
        ['k', ]
    ),
}


def new(abbr, *args, mesh=None, **kwargs):
    """Generate a spaces on the mesh.

    Parameters
    ----------
    abbr : str
        The abbr. of the space.
    args :
        The arguments to be sent to the space.
    mesh : {:class:`Mesh`, None}, optional
        We want to generate space on this mesh. If it is ``None``, we use the current target mesh.
        The default value is ``None``.
    kwargs :
        The keyword arguments to be sent to the space.

    Returns
    -------
    space :
        The finite dimensional space.

    """
    if _config['current_mesh'] == '' and mesh is None:
        raise Exception(f"pls set a mesh firstly by using 'space.set_mesh' or specify 'mesh'.")
    else:
        pass

    if isinstance(abbr, str):  # make only 1 space
        pass
    else:
        raise NotImplementedError()

    mesh_sr = _config['current_mesh']

    if mesh is None:
        mesh = _mesh_set[mesh_sr]
    else:
        # noinspection PyUnresolvedReferences
        mesh_sr = mesh._sym_repr
        if mesh_sr in _mesh_set:
            pass
        else:
            _mesh_set[mesh_sr] = mesh
            _space_set[mesh_sr] = dict()

    current_spaces = _space_set[mesh_sr]

    assert abbr in _implemented_spaces, \
        f"space abbr.={abbr} not implemented. do 'ph.space.list_()' to see all implemented spaces."

    space_class_path, space_class_name = _implemented_spaces[abbr][0:2]

    space_class = getattr(import_module(space_class_path), space_class_name)

    space = space_class(mesh, *args, **kwargs)

    srp = space._sym_repr  # do not use __repr__()

    if srp in current_spaces:
        pass
    else:
        current_spaces[srp] = space

    space = current_spaces[srp]

    return space


__all__ = [
    '_VarSetting_mass_matrix',                     #
    '_VarSetting_trace_matrix',                    # trace matrix
    '_VarSetting_d_matrix',                        #
    '_VarSetting_d_matrix_transpose',              #
    '_VarSetting_pi_matrix',
    '_VarSetting_star_matrix',                     # Hodge matrix
    '_VarSetting_dp_matrix',                       # <A|B>

    '_VarSetting_boundary_dp_vector',              #

    '_VarSetting_astA_convect_astB_ip_tC',         # (*A .V *B, @C), AB are known, vector.

    '_VarSetting_astA_x_astB_ip_tC',               #
    '_VarSetting_astA_x_B_ip_tC',                  #
    '_VarSetting_A_x_astB_ip_tC',                  #
    '_VarSetting_A_x_B_ip_C',                      # nonlinear

    '_VarSetting_astA_x_astB__dp__tC',             # vector <*A x *B | @C>, AB are known
    '_VarSetting_astA_x_B__dp__tC',
    '_VarSetting_A_x_astB__dp__tC',
    '_VarSetting_A_x_B__dp__C',                      # nonlinear

    '_VarSetting_astA_x_astB__ip__astC_x_tD',      # vector  (*A x *B, *C x @D), ABC known, D test.
    '_VarSetting_A_x_astB__ip__astC_x_tD',         # matrix  (A x *B, *C x @D), BC known, D test.

    '_VarSetting_astA_x_astB__dp__astC_x_tD',      # vector  <*A x *B | *C x @D>, ABC known, D test.

    # (A, BC) ---------------------------------------------------------------------
    '_VarSetting_A_ip_BC',                         # (A, BC); nonlinear; A, B, C all unknown
    '_VarSetting_astA_ip_B_tC',                    # (A, BC); linear, A given, matrix, C test form
    '_VarSetting_A_ip_astB_tC',                    # (A, BC); linear, B given, matrix, C test form
    '_VarSetting_astA_ip_astB_tC',                 # (A, BC); A and B given, vector, C test form

    # <AB|C> ---------------------------------------------------------------------
    '_VarSetting_AB_dp_C',                         # <AB|C> nonlinear; A, B, C all unknown
    '_VarSetting_astA_B_dp_tC',                    # <AB|C>; linear, A given, matrix, C test form
    '_VarSetting_A_astB_dp_tC',                    # <AB|C>; linear, B given, matrix, C test form
    '_VarSetting_astA_astB_dp_tC',                 # <AB|C>; A and B given, vector, C test form

    # <A d(B)|C> ---------------------------------------------------------------------
    '_VarSetting_A_d_B_dp_C',         # <A d(B)|C> nonlinear; A, B, C all unknown
    '_VarSetting_astA_d_B_dp_tC',     # <A d(B)|C>; linear, A given, matrix, C test form
    '_VarSetting_A_d_astB_dp_tC',     # <A d(B)|C>; linear, B given, matrix, C test form
    '_VarSetting_astA_d_astB_dp_tC',  # <A d(B)|C>; A and B given, vector, C test form

    # (AB, C) ---------------------------------------------------------------------
    '_VarSetting_AB_ip_C',                         # (AB, C) nonlinear; A, B, C all unknown
    '_VarSetting_astA_B_ip_tC',                    # (AB, C); linear, A given, matrix, C test form
    '_VarSetting_A_astB_ip_tC',                    # (AB, C); linear, B given, matrix, C test form
    '_VarSetting_astA_astB_ip_tC',                 # (AB, C); A and B given, vector, C test form

    # (AB, dC) ---------------------------------------------------------------------
    '_VarSetting_AB_ip_dC',                         # (AB, dC) nonlinear; A, B, C all unknown
    '_VarSetting_astA_B_ip_dtC',                    # (AB, dC); linear, A given, matrix, C test form
    '_VarSetting_A_astB_ip_dtC',                    # (AB, dC); linear, B given, matrix, C test form
    '_VarSetting_astA_astB_ip_dtC',                 # (AB, dC); A and B given, vector, C test form

    # (A d(B), d(C)) ------------------------------------------------------------------
    '_VarSetting_A_d_B_ip_dC',                      # (A d(B), dC) nonlinear; A, B, C all unknown
    '_VarSetting_astA_d_B_ip_dtC',                  # (A d(B), dC) linear, A given, matrix, C test form
    '_VarSetting_A_d_astB_ip_dtC',                  # (A d(B), dC) linear, B given, matrix, C test form
    '_VarSetting_astA_d_astB_ip_dtC',               # (A d(B), dC) A and B given, vector, C test form

    # <AB|dC> ---------------------------------------------------------------------
    '_VarSetting_AB_dp_dC',                         # <AB|dC> nonlinear; A, B, C all unknown
    '_VarSetting_astA_B_dp_dtC',                    # <AB|dC>; linear, A given, matrix, C test form
    '_VarSetting_A_astB_dp_dtC',                    # <AB|dC>; linear, B given, matrix, C test form
    '_VarSetting_astA_astB_dp_dtC',                 # <AB|dC>; A and B given, vector, C test form

    # ------ MORE NONLINEAR TERMS ----------------------------------------------------
    '_VarSetting_log_e_astA_ip_tB',                 # (log_e A, B) where A is given and B is the test form

    # =================================================================================

    # bundle valued forms ---------------------------------------------------------
    '_VarSetting_dastA_astA_tp_tC',                #
    '_VarSetting_dastA_tB_tp_astA',
    '_VarSetting_dtA_astB_tp_astB',
    '_VarSetting_dA_B_tp_C__1Known',
    '_VarSetting_dA_B_tp_C__2Known',
    '_VarSetting_dA_B_tp_C',                       # nonlinear
    '_VarSetting_AxB_ip_dC',                        # nonlinear

    '_VarSetting_A_B_tp_C__1Known',
    '_VarSetting_A_B_tp_C__2Known',
    '_VarSetting_A_B_tp_C',                       # nonlinear

    '_VarSetting_IP_matrix_db_bf',                #
    '_VarSetting_IP_matrix_bf_db',                #

    # '_VarSetting_A_x_astB_ip_dC',                 # (A x B, dC)
    # '_VarSetting_astA_x_B_ip_dC',                 # (A x B, dC)
    # '_VarSetting_astA_x_astB_ip_dC',              # (A x B, dC)
]


# ------ basic -----------------------------------------------------------------------------------
_VarSetting_mass_matrix = [
    r"\mathsf{M}",
    _sep.join(["Mass:Mat", "{space_pure_lin_repr}", "{d0}", "{d1}"]),
]

_VarSetting_trace_matrix = [
    r"\mathbb{T}",
    _sep.join(["Trace:Mat", "{space_pure_lin_repr}", "{degree}"]),
]

# _VarSetting_trace_mass_matrix = [
#     r"\mathbb{M}",
#     _sep.join(["Trace:Mat", "{space_pure_lin_repr}", "{degree}"]),
# ]

_VarSetting_d_matrix = [
    r"\mathsf{D}",
    _sep.join(["d:Mat", "{space_pure_lin_repr}", "{d}"]),
]

_VarSetting_d_matrix_transpose = [
    r"\mathbb{D}",
    _sep.join(["d:T:Mat", "{space_pure_lin_repr}", "{d}"]),
]

_VarSetting_pi_matrix = [
    r"\mathsf{P}",
    _sep.join([
        "d:P:Mat",
        "{space_pure_lin_repr_from}", "{space_pure_lin_repr_to}",
        "{d_from}", "{d_to}"  # degree_from, degree_to
    ]),
]

_VarSetting_star_matrix = [
    r"\mathsf{H}",
    _sep.join([
        "Hodge:Mat",
        "{space_pure_lin_repr_from}", "{space_pure_lin_repr_to}",
        "{d_from}", "{d_to}"   # degree_from, degree_to
    ]),
]

_VarSetting_dp_matrix = [   # <A|B> or <B|A> : 0 refers to the axis-0 space.
    r"\mathsf{W}",
    _sep.join(["Wedge:Mat", "{s0}", "{s1}", "{d0}", "{d1}"]),
]

# Natural bc -------------------------------------------------------------------------------------

_VarSetting_boundary_dp_vector = [
    # once we know f0, we can find the correct basis functions it wedged with
    r"\boldsymbol{b}",
    _sep.join(["BoundaryDP:Vec", "trStar[{f0}]", "tr[{f1}]"]),
    #                            <tr star bf0 | tr f1>.
]

# (A .V B, C) -------------------------------------------------------------------------------------
_VarSetting_astA_convect_astB_ip_tC = [
    r"\mathsf{V}_{\left({A} \cdot\nabla {B}, \mathsf{t}\right)}^{\left[\mathsf{t}\right]}",
    _sep.join(["*convect*_ip", "[{A}]", "[{B}]", "[{C}]"]),
]

# (w x u, u) --------------------------------------------------------------------------------------

_VarSetting_astA_x_astB_ip_tC = [
    r"\mathsf{V}_{\left({A}\times{B}, \mathsf{t}\right)}^{\left[\mathsf{t}\right]}",
    _sep.join(["c_ip", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_astA_x_B_ip_tC = [
    r"\mathsf{M}_{\left({A}\times \circ, \mathsf{t}\right)}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["X_ip", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_A_x_astB_ip_tC = [
    r"\mathsf{M}_{\left(\circ\times {B}, \mathsf{t}\right)}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["_Xip", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_A_x_B_ip_C = [
    r"\left(\mathsf{\cdot x\cdot},\cdot\right)",
    _sep.join(["_X_:", "[{A}]", "[{B}]", "[{C}]"]),
]

# (A, BC) ---------------------------------------------------------------------------------------
_VarSetting_A_ip_BC = [               # nonlinear
    r"\left(\cdot,\cdot \ast \cdot\right)",
    _sep.join(["_;_*_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_astA_ip_B_tC = [         # A given; C test form; matrix
    r"\mathsf{M}_{\left({A},\cdot \ast \mathsf{t}\right)}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["X;_*_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_A_ip_astB_tC = [         # B given; C test form; matrix
    r"\mathsf{M}_{\left(\cdot,{B} \ast \mathsf{t}\right)}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["_;X*_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_astA_ip_astB_tC = [         # A and B given; C test form; vector
    r"\mathsf{V}_{\left({A},{B} \ast \mathsf{t}\right)}^{\left[\mathsf{t}\right]}",
    _sep.join(["X;X*_:", "[{A}]", "[{B}]", "[{C}]"]),
]

# <AB|C> -----------------------------------------------------------------------------------------
_VarSetting_AB_dp_C = [               # nonlinear
    r"\left<\left.\cdot \ast \cdot \right| \cdot\right>",
    _sep.join(["_*_|_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_astA_B_dp_tC = [         # A given; C test form; matrix
    # r"M{A}",
    r"\mathsf{M}_{\left<\left. {A} \ast \circ \right|\mathsf{t}\right>}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["X*_|_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_A_astB_dp_tC = [         # B given; C test form; matrix
    r"\mathsf{M}_{\left<\left.\circ \ast {B} \right|\mathsf{t}\right>}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["_*X|_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_astA_astB_dp_tC = [         # A and B given; C test form; vector
    r"\mathsf{V}_{\left<\left. {A} \ast {B} \right|\mathsf{t}\right>}^{\left[\mathsf{t}\right]}",
    _sep.join(["X*X|_:", "[{A}]", "[{B}]", "[{C}]"]),
]

# <A d(B)|C> -----------------------------------------------------------------------------------------
_VarSetting_A_d_B_dp_C = [               # nonlinear
    r"\left<\left.\cdot \ast \mathrm{d}\cdot \right| \cdot\right>",
    _sep.join(["_*d_|_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_astA_d_B_dp_tC = [         # A given; C test form; matrix
    # r"M{A}",
    r"\mathsf{M}_{\left<\left. {A} \ast \mathrm{d} \circ \right|\mathsf{t}\right>}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["X*d_|_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_A_d_astB_dp_tC = [         # B given; C test form; matrix
    r"\mathsf{M}_{\left<\left.\circ\mathrm{d} \ast {B} \right|\mathsf{t}\right>}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["_*dX|_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_astA_d_astB_dp_tC = [         # A and B given; C test form; vector
    r"\mathsf{V}_{\left<\left. {A} \ast \mathrm{d} {B} \right|\mathsf{t}\right>}^{\left[\mathsf{t}\right]}",
    _sep.join(["X*dX|_:", "[{A}]", "[{B}]", "[{C}]"]),
]

# (AB, C) -----------------------------------------------------------------------------------------
_VarSetting_AB_ip_C = [               # nonlinear
    r"\left(\cdot \ast \cdot,\cdot\right)",
    _sep.join(["_*_,_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_astA_B_ip_tC = [         # A given; C test form; matrix
    r"\mathsf{M}_{\left({A} \ast \circ ,\mathsf{t}\right)}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["X*_,_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_A_astB_ip_tC = [         # B given; C test form; matrix
    r"\mathsf{M}_{\left( \circ \ast {B} ,\mathsf{t}\right)}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["_*X,_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_astA_astB_ip_tC = [         # A and B given; C test form; vector
    r"\mathsf{V}_{\left( {A} \ast {B} ,\mathsf{t}\right)}^{\left[\mathsf{t}\right]}",
    _sep.join(["X*X,_:", "[{A}]", "[{B}]", "[{C}]"]),
]

# (AB, d(C)) -----------------------------------------------------------------------------------------
_VarSetting_AB_ip_dC = [               # nonlinear
    r"\left(\cdot \ast \cdot,\mathrm{d}\cdot\right)",
    _sep.join(["_*_,d_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_astA_B_ip_dtC = [         # A given; C test form; matrix
    # r"M{A}",
    r"\mathsf{M}_{\left({A} \ast \circ ,\mathrm{d}\mathsf{t}\right)}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["X*_,d_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_A_astB_ip_dtC = [         # B given; C test form; matrix
    r"\mathsf{M}_{\left( \circ \ast {B} ,\mathrm{d}\mathsf{t}\right)}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["_*X,d_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_astA_astB_ip_dtC = [         # A and B given; C test form; vector
    r"\mathsf{V}_{\left( {A} \ast {B} ,\mathrm{d}\mathsf{t}\right)}^{\left[\mathsf{t}\right]}",
    _sep.join(["X*X,d_:", "[{A}]", "[{B}]", "[{C}]"]),
]

# (A d(B), d(C)) -----------------------------------------------------------------------------------------
_VarSetting_A_d_B_ip_dC = [               # nonlinear
    r"\left(\cdot \ast \mathrm{d}\cdot,\mathrm{d}\cdot\right)",
    _sep.join(["_*d_,d_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_astA_d_B_ip_dtC = [         # A given; test form C; matrix
    # r"M{A}",
    r"\mathsf{M}_{\left({A} \ast \mathrm{d}\circ ,\mathrm{d}\mathsf{t}\right)}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["X*d_,d_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_A_d_astB_ip_dtC = [         # B given; test form C; matrix
    r"\mathsf{M}_{\left( \circ \ast \mathrm{d} {B} ,\mathrm{d}\mathsf{t}\right)}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["_*dX,d_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_astA_d_astB_ip_dtC = [         # A and B given; test form C; vector
    r"\mathsf{V}_{\left( {A} \ast \mathrm{d} {B} ,\mathrm{d}\mathsf{t}\right)}^{\left[\mathsf{t}\right]}",
    _sep.join(["X*dX,d_:", "[{A}]", "[{B}]", "[{C}]"]),
]


# <AB|dC> -----------------------------------------------------------------------------------------
_VarSetting_AB_dp_dC = [               # nonlinear
    r"\left<\left.\cdot \ast \mathrm{d}\cdot\right|\cdot\right>",
    _sep.join(["_*_|d_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_astA_B_dp_dtC = [         # A given; C test form; matrix
    r"\mathsf{M}_{\left<{A}\left. \ast \circ\right|\mathrm{d}\mathsf{t}\right>}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["X*_|d_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_A_astB_dp_dtC = [         # B given; C test form; matrix
    r"\mathsf{M}_{\left<\left. \circ \ast {B}\right|\mathrm{d}\mathsf{t}\right>}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["_*X|d_:", "[{A}]", "[{B}]", "[{C}]"]),
]

_VarSetting_astA_astB_dp_dtC = [         # A and B given; C test form; vector
    r"\mathsf{V}_{\left< \left.{A} \ast {B}\right|\mathrm{d}\mathsf{t}\right>}^{\left[\mathsf{t}\right]}",
    _sep.join(["X*X|d_:", "[{A}]", "[{B}]", "[{C}]"]),
]

# -------- MORE NONLINEAR TERMS -------------------------------------------------------------------


_VarSetting_log_e_astA_ip_tB = [
    r"\mathsf{V}_{\left(\ln {A}, \mathsf{t}\right)}",
    _sep.join(["log_e_*_ip__:", "[{A}]", "[{B}]"]),
]


# --------------- <A x B | C> --------------------------------------------------------------------
_VarSetting_astA_x_astB__dp__tC = [   # <*A x *B | @D>
    r"\mathsf{V}_{\left\langle\left.{A}\times {B} \right| \mathsf{t}\right\rangle}^{\left[\mathsf{t}\right]}",
    _sep.join(["<*x*|C>", "[{A}]", "[{B}]", "[{C}]"])
]

_VarSetting_astA_x_B__dp__tC = [   # <*A x B | @D>
    r"\mathsf{M}_{\left\langle\left.{A}\times \circ \right| \mathsf{t}\right\rangle}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["<*xB|C>", "[{A}]", "[{B}]", "[{C}]"])
]

_VarSetting_A_x_astB__dp__tC = [   # <A x *B | @D>
    r"\mathsf{M}_{\left\langle\left. \circ \times {B} \right| \mathsf{t}\right\rangle}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["<Ax*|C>", "[{A}]", "[{B}]", "[{C}]"])
]

_VarSetting_A_x_B__dp__C = [
    r"\left<\left.\mathsf{\cdot x\cdot}\right|\cdot\right>",
    _sep.join(["_XD_:", "[{A}]", "[{B}]", "[{C}]"]),
]

# --------------- (A x B, C x D) ----------------------------------------------------
_VarSetting_astA_x_astB__ip__astC_x_tD = [   # (*A x *B, *C x @D)
    r"\mathsf{V}_{\left({A} \times {B}, {C} \times \mathsf{t}\right)}^{\left[\mathsf{t}\right]}",
    _sep.join(["(*x*,*xD)", "[{A}]", "[{B}]", "[{C}]", "[{D}]"]),
]

_VarSetting_A_x_astB__ip__astC_x_tD = [    # (A x *B, *C x @D)
    r"\mathsf{M}_{\left(\circ \times {B}, {C} \times \mathsf{t}\right)}^{\left[\mathsf{t},\circ\right]}",
    _sep.join(["(Ax*,*xD)", "[{A}]", "[{B}]", "[{C}]", "[{D}]"]),
]

# --------------- <A x B | C x D> ----------------------------------------------------
_VarSetting_astA_x_astB__dp__astC_x_tD = [  # <*A x *B | *C x @D>
    r"\mathsf{V}_{\left\langle\left.{A}\times {B} \right| {C}\times \mathsf{t}\right\rangle}^{\left[\mathsf{t}\right]}",
    _sep.join(["<*x*|*xD>", "[{A}]", "[{B}]", "[{C}]", "[{D}]"]),
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

# (AxB, dc): nonlinear --------------------------------------------

_VarSetting_AxB_ip_dC = [  # A, B, C are different; # nonlinear
    r"\left(\mathsf{\cdot x\cdot}, \mathsf{d}\cdot\right)",
    _sep.join(["AxB_ip_dC", "[{A}]", "[{B}]", "[{C}]"]),
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
    from phyem.src.config import RANK, MASTER_RANK
    if RANK != MASTER_RANK:
        return
    else:
        pass

    print('Implemented spaces:')
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
