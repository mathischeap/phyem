# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
r"""

.. note::

    *msepy* stands for **mimetic spectral elements in Python**. It is an implementation of
    the mimetic spectral element methods using pure Python. In other words, the most
    computationally intensive part of the simulation, the linear system solving, is also
    done within Python.

    **Advantages** of this are clear. No external packages (or APIs to other kernels)
    are needed to use *msepy* implementation; only common Python packages like scipy, numpy,
    matplotlib, etc., are required. Users can quickly settle *phyem* in their machines
    no mather they are Windows, Linux or Mac.

    The most obvious **disadvantage** of this implementation is that, since Python suffers from
    its relatively lower speed, this implementation is not proper for large problems. It is
    hard to say what problems are large. And it is also dependent on the machine. Normally,
    *msepy* is very handy for 1- or 2-dimentional and small 3-dimensional problems. User could
    explore the edge by trial.

.. caution::

    *msepy* is not parallelizable; do not execuate with for example ``mpiexec``.

To invoke the msepy implementation, use indicator ``'msepy'`` as the
first argument for ``apply`` of ``fem`` module, i.e.,

>>> implementation, objects = ph.fem.apply('msepy', locals())

We get the implementation body, ``implementation``, and the dictionary of all
implemented counterparts, ``objects``.


.. _Implementations-msepy-counterparts:

Implemented counterparts
************************

To pick up the implemented counterparts of abstract instances, we can just use their variable names, for example,

>>> manifold = objects['manifold']
>>> mesh = objects['mesh']

If some instances have no explicit varialbes, you could possiblly
pick them up using their symbolic representations.
For example, the boundary manifolds for defineing the boundary conditions have no explicit varibles.
We can pick them using theire symbolic representations through the dictionary ``implementation.base``,

>>> Gamma_alpha = implementation.base['manifolds'][r"\Gamma_{\alpha}"]
>>> Gamma_beta = implementation.base['manifolds'][r"\Gamma_{\beta}"]

For these instances that can be accessed throug ``implementation.base``,
there are just four types, manifolds, meshes, spaces and forms. They can be accessed
with keys ``'manifolds'``, ``'meshes'``, ``'spaces'`` and ``'forms'``,
respectively. For example

>>> mesh is implementation.base['meshes'][r'\mathfrak{M}']
True

We see that the mesh accessed from ``implementation.base['meshes']`` is the same mesh we obtained
from ``objects``.


.. _Implementations-msepy-config:

Configuration
*************

❇️ **Manifold & Mesh**

There are two ways to configure the abstract manifold and mesh to be exact ones:

*1) Use predefined setting*:

*msepy* has predefined maniolds (domains). To let a manifold to be a predefined one,
we can call ``config`` of the implementation body, ``implementation``, for example,

>>> implementation.config(manifold)(
...     'crazy', c=0., bounds=([0, 1], [0, 1]), periodic=False,
... )

where the first argument, ``'crazy'``, is the indicator and the remaining arguments are
parameters of the predefined manifold.

.. admonition:: Predefined *msepy* manifolds

    +------------------------+---------------------------------------------------------------------+
    | **indicator**          | **description**                                                     |
    +------------------------+---------------------------------------------------------------------+
    | ``'crazy'``            | See :ref:`GALLERY-msepy-domains-and-meshes=crazy`.                  |
    +------------------------+---------------------------------------------------------------------+
    | ``'crazy_multi'``      | See :ref:`GALLERY-msepy-domains-and-meshes=multi-crazy`.            |
    +------------------------+---------------------------------------------------------------------+
    | ``'backward_step'``    | See :ref:`GALLERY-msepy-domains-and-meshes=backward-step`.          |
    +------------------------+---------------------------------------------------------------------+
    | ``'cylinder_channel'`` | See :ref:`GALLERY-msepy-domains-and-meshes=cylinder-channel`        |
    +------------------------+---------------------------------------------------------------------+
    | ...                    |                                                                     |
    |                        |                                                                     |
    |                        |                                                                     |
    +------------------------+---------------------------------------------------------------------+

All predefined *msepy* manifolds are available at Gallery, see :ref:`GALLERY-msepy-domains-and-meshes`.

Once we have specified the manifold, we should also configure its boundary sections, for example,

>>> implementation.config(Gamma_alpha)(
...     manifold, {0: [1, 1, 0, 0]}   # the natural boundary.
... )

This specifies the boundary section ``Gamma_alpha``, :math:`\Gamma_{\alpha}`, to be two faces of the *msepy*
manifold ``manifold``. These faces are indicated by dictionary ``{0: [1, 1, 0, 0]}``. For what does this
dictionary exactly mean, we refer to the introduction page of the ``'crazy'`` domain,
:ref:`GALLERY-msepy-domains-and-meshes=crazy`. In short, the manifold is one region
(or divided into multiple regions) with a local coordinate system :math:`(r, s)`, and
``{0: [1, 1, 0, 0]}`` indicates the topological faces
facing :math:`r-` and :math:`r+` directions of region no. ``0``.

.. note::
    Since ``Gamma_alpha`` and ``Gamma_beta`` are a partition of the boundary of ``manifold``,
    see :ref:`PDE-bc`, once ``Gamma_alpha`` is configured, ``Gamma_beta`` is definite; no need
    to configure it.

Once the manifold and its boundary sections are configured, we can configure the mesh on it by, for example,

>>> implementation.config(mesh)([12, 12])

This will generate a mesh of :math:`12 \times 12` elements in the manifold. Both ``manifold`` and ``mesh``
can be visualized by calling their ``visualize`` property.
See examples at :ref:`GALLERY-msepy-domains-and-meshes`.

*2) User-customized meshes*:

.. todo::

    Users shall also could input their own meshes (thus maniolds are definite) generated in common
    mesh generators, for instance,
    `Gmsh <https://gmsh.info/>`_.
    To this end, an interface from the
    generic mesh format, for example, the
    `VTK <https://vtk.org/>`_
    format, to *msepy* mesh and manifold modules should
    be implemented.

    So far, this interface is missing. And if your domain is not covered by the predefined manifolds, please
    send us a message, see :ref:`contact`, with your domain details. We will implement it as a
    predefined manifold as soon as possible.

|

❇️ **Time sequence**

We also need to specify the abstract time sequence to an exact one. Since the time sequence instance, ``ts``,
is not generic regardless of particular implementations, we do not need to pick it up from the implementation.

>>> ts.specify('constant', [0, 1, 100], 2)

This command specifies the time sequence ``ts`` to be one of constant
(indicated by ``'constant'``) time intervals, i.e.,

.. math::
    \Delta t = t^{k} - t^{k-1} = C\quad \forall k\in\left\lbrace 1, 2, 3,\cdots\right\rbrace.

The time sequence starts with :math:`t=0`, ends with :math:`t=1`, and places 100 equal smallest intervals
in between, and each time step covers two smallest intervals.
Thus, for example, the time step from :math:`t^{k-1}` to :math:`t^{k}` is splitted into two equal smallest
intervals, i.e., one from :math:`t^{k-1}` to :math:`t^{k-\frac{1}{2}}` and
one from :math:`t^{k-1}` to :math:`t^{k}`. In other words, valid time instances of this time sequence are

.. math::
    t^0,\ t^{\frac{1}{2}},\ t^1,\ t^{1+\frac{1}{2}},\ \cdots,\ t^{49+\frac{1}{2}},\ t^{50}.

And in total, we have 50 time steps, :math:`k\in\left\lbrace1,2,3,\cdots,50\right\rbrace`.

|

❇️ **Initial condition**

Suppose the linear port Hamiltonian problem we try to solve has an analytic solution,

.. math::
    \left\lbrace
    \begin{aligned}
        & \tilde{\alpha}_{\mathrm{analytic}}(x, y, t) = - g(x,y)\dfrac{\mathrm{d}f(t)}{\mathrm{d}t} \\
        & \tilde{\beta}_{\mathrm{analytic}}(x, y, t) = \begin{bmatrix}
            \dfrac{\partial g}{\partial x}(x, y) f(t) \\
            \dfrac{\partial g}{\partial y}(x, y) f(t)
        \end{bmatrix}
    \end{aligned}
    \right.,

where

.. math::
    g(x, y) = \cos(2\pi x)\cos(2\pi y),

and

.. math::
    f(t) = 2\sin(2\sqrt{2}\pi) + 3\cos(2\sqrt{2}\pi).

This is a so-called 2-dimensional eigen solution. And it is pre-coded in *phyem*. We can call it by

>>> eigen2 = ph.samples.Eigen2()

And the analytic solutions, :math:`\tilde{\alpha}_{\mathrm{analytic}}`
and :math:`\tilde{\beta}_{\mathrm{analytic}}`, are attributes, ``scalar`` and ``vector``, of ``eigen2``.
We can set the continuous form of ``a`` and ``b`` to be :math:`\tilde{\alpha}_{\mathrm{analytic}}`
and :math:`\tilde{\beta}_{\mathrm{analytic}}` by

>>> a = objects['a']
>>> b = objects['b']
>>> a.cf = eigen2.scalar
>>> b.cf = eigen2.vector

where the first two lines pick up the implemented counterparts of forms ``a`` and ``b`` and the last two
lines set their continuous forms, ``cf``, to be the analytic solutions. If a form has its continuous form,
we can measure the error between its numerical solution to its continuous form (analytic solution)
which indicates the accuaracy of
the simulation. For example,

>>> a[0].reduce()                      # reduce the analytic solution at t=0 to discrete space
>>> b[0].reduce()                      # reduce the analytic solution at t=0 to discrete space
>>> a_L2_error_t0 = a[0].error()       # compute the L2 error at t=0
>>> b_L2_error_t0 = b[0].error()       # compute the L2 error at t=0
>>> a_L2_error_t0
0.0056...
>>> b_L2_error_t0
0.0060...

.. note::

    The brackets, ``[]``, of a form return a staic copy of the form at a particular time instant.
    For example, ``a[0]`` gives
    the static copy of ``a`` at :math:`t=0`.

So, the first two lines of above code discretize the analytic solution at :math:`t=0` to the discrete forms
``a`` and ``b``, which
configures the initial condition of the simulation. The next two lines then compute the error between the discrete
initial condition and the analytic initial condition. It is seen that the error is very low implying that
the finite dimensional spaces in this mesh (of :math:`12\times12` elements) are fine.

Note that a generic simulation usually does not possess an analytical solution for all time, but only has the initial
condition. In that case, we can use the same way to initialize the discrete forms to possess the initial condition.

|

❇️ **Boundary conditions**

To impose specified boundary conditions, we need to touch the counterpart of the linear system. Thus, we first
need to pick up the implemented linear system by its local variable name, i.e.,

>>> ls = objects['ls']

And to let it take effect, we need to call its ``apply`` method,

>>> ls = ls.apply()

Then we can configure the boundary conditions of this particular linear system by using ``config`` of its property
``bc``,

>>> ls.bc.config(Gamma_alpha)(eigen2.scalar)   # natural boundary condition
>>> ls.bc.config(Gamma_beta)(b.cf)             # essential boundary condition

where the first command sets the natural boundary condition on :math:`\Gamma_\alpha` to be the analytical
solution :math:`\tilde{\alpha}_{\mathrm{analytic}}` and the second line sets the essnetial boundary condition
on :math:`\Gamma_\beta` to be the analytical
solution :math:`\tilde{\beta}_{\mathrm{analytic}}` (recall that we have set ``b.cf`` to be
``eigen2.vector``).

.. _Implementations-msepy-solving:

Solving
*******

We first define two lists to store the errors,

>>> a_errors = [a_L2_error_t0, ]
>>> b_errors = [b_L2_error_t0, ]

We then go through all time steps by iterating over all :math:`k\in\left\lbrace1,2,3,\cdots,50\right\rbrace`,

>>> for k in range(1, 51):
...     static_ls = ls(k=k)                      # get the static linear system for k=...
...     assembled_ls = static_ls.assemble()      # assemble the static linear system into a global system
...     x, message, info = assembled_ls.solve()  # solve the global system
...     static_ls.x.update(x)                    # use the solution to update the discrete forms
...     a_L2_error = a[None].error()             # compute the error of the discrete forms at the most recent time
...     b_L2_error = b[None].error()             # compute the error of the discrete forms at the most recent time
...     a_errors.append(a_L2_error)              # append the error to the list
...     b_errors.append(b_L2_error)              # append the error to the list

Note that ``a[None]`` automatically gives a static copy of ``a`` at its most recent time. For example, when
``k=10``, i.e., :math:`t=t^{10}=0.2` (recall that the overall computation time is 1 and it is divided into
50 time steps), ``a[None]`` is equivalent to ``a[0.2]``. And then method ``error`` computes the :math:`L^2`-error of
it. To check it, do

>>> a[0.2].error() == a_errors[10]
True
>>> len(a_errors)
51
>>> len(b_errors)
51


.. _Implementations-msepy-pp:

Post-processing
***************

You can save the solution to VTK file by calling

>>> a[1].visualize.vtk(saveto='a1')
>>> b[1].visualize.vtk(saveto='b1')

The first line saves the static copy of ``a`` at :math:`t=1` to ``./a1.vtu`` and the second line
saves the static copy of ``b`` at :math:`t=1` to ``./b1.vtu``.

You can also save them into one file by

>>> a[1].visualize.vtk(b[1], saveto='a1_b1')

Static copies of ``a`` and ``b`` at :math:`t=1` are saved to ``./a1_b1.vtu``.

Then visualization tools can be used to visualize and analyze the solutions. And we recommend,
for example, the open-source
visualization software `Paraview <https://www.paraview.org/>`_.

You can also try

.. code-block::

    a[1].visualize.matplot()

This will give a 2-dimensioan plot of ``a[1]`` using the matplotlib package.
But since matplotlib is not very handy for 3-dimensional plots,
the ``matplot`` method is only implemented for 2-dimensional forms.
And we again recommand interactive VTK visualization tools for 3-dimensional visualization.

"""

from tools.frozen import Frozen
from msepy.manifold.main import MsePyManifold
from msepy.mesh.main import MsePyMesh
from msepy.space.main import MsePySpace
from msepy.form.main import MsePyRootForm
from src.wf.mp.linear_system import MatrixProxyLinearSystem
from src.wf.mp.nonlinear_system import MatrixProxyNoneLinearSystem
from src.config import SIZE   # MPI.SIZE
from time import time
from tools.miscellaneous.timer import MyTimer
from msepy.tools.gathering_matrix import _cgm_cache


__setting__ = {}


__all__ = [
    '_parse_manifolds',
    '_parse_meshes',
    '_parse_spaces',
    '_parse_root_forms',
    '_parse',

    'config',
    "new",
]


# will be cleared whenever new msepy implementation is called.
base = {
    'manifolds': dict(),  # keys: abstract manifold sym_repr
    'meshes': dict(),     # keys: abstract mesh sym_repr
    'spaces': dict(),  # keys: abstract space sym_repr, values: MsePy spaces
    'forms': dict(),  # keys: abstract root form pure_lin_repr, values: root-forms,
}


_info_cache = {
    'start_time': -1.,
    'info_count': 0,
    'info_time': -1.,
}


def _check_config():
    """Check the configuration is compatible or not."""
    assert SIZE == 1, f"msepy only works for single thread call (MPI.SIZE=1), now MPI.size = {SIZE}"


def _clear_self():
    """Clear self to make sure the previous implementation does not mess things up!"""
    base['manifolds'] = dict()
    base['meshes'] = dict()
    base['spaces'] = dict()
    base['forms'] = dict()
    _info_cache['start_time'] = -1.
    _info_cache['info_count'] = 0
    _info_cache['info_time'] = -1.
    _cgm_cache['signatures'] = ''
    _cgm_cache['cgm'] = None


def _parse_manifolds(abstract_manifolds):
    """"""
    manifold_dict = {}
    for sym in abstract_manifolds:
        manifold = MsePyManifold(abstract_manifolds[sym])
        manifold_dict[sym] = manifold
    base['manifolds'] = manifold_dict


def _parse_meshes(abstract_meshes):
    """"""
    mesh_dict = {}
    for sym in abstract_meshes:
        am = abstract_meshes[sym]
        m = MsePyMesh(am)
        mesh_dict[sym] = m
    base['meshes'] = mesh_dict


def _parse_spaces(abstract_spaces):
    """"""
    space_dict = {}
    for ab_msh_sym_repr in abstract_spaces:
        ab_sps = abstract_spaces[ab_msh_sym_repr]

        for ab_sp_sym_repr in ab_sps:
            ab_sp = ab_sps[ab_sp_sym_repr]

            if ab_sp.orientation != 'unknown':  # Those spaces are probably not for root-forms, skipping is OK.
                space = MsePySpace(ab_sp)
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
            prf = MsePyRootForm(rf)
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
            prf = MsePyRootForm(rf)
            prf._pAti_form['base_form'] = particular_base_form
            prf._pAti_form['ats'] = ats
            prf._pAti_form['ati'] = ati
            rf_dict[pure_lin_repr] = prf

            assert rf_lin_repr not in particular_base_form._ats_particular_forms
            particular_base_form._ats_particular_forms[rf_lin_repr] = prf

    for pure_lin_repr in rf_dict:
        assert rf_dict[pure_lin_repr].degree is not None, \
            f"must be, an abstract root of no degree cannot be implemented."

    base['forms'] = rf_dict


def _parse(obj):
    """The objects other than manifolds, meshes, spaces, root-forms that should be parsed for this
    particular fem setting.
    """
    if obj.__class__ is MatrixProxyLinearSystem:
        from msepy.tools.linear_system.dynamic.main import MsePyDynamicLinearSystem
        dynamic = MsePyDynamicLinearSystem(obj, base)
        return dynamic
    elif obj.__class__ is MatrixProxyNoneLinearSystem:
        from msepy.tools.nonlinear_system.dynamic.main import MsePyDynamicNonLinearSystem
        dynamic = MsePyDynamicNonLinearSystem(obj, base)
        return dynamic

    else:
        return None  # do not raise Error (like below)!
        # raise NotImplementedError(f"cannot parse msepy implementation for {obj}.")


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
    print(f'=== [{count}] {MyTimer.current_time()} -after- %.2f(s)'
          f', total: {MyTimer.seconds2dhms(total_cost)} <----' % (new_time - old_time))
    print(f"~) Form with newest cochain @ --------- ")
    for form_sym in forms:
        form = forms[form_sym]
        if form._is_base():
            newest_time = form.cochain.newest
            if newest_time is not None:
                print('{:>20} @ {:<30}'.format(form.abstract._pure_lin_repr, newest_time))
        else:
            pass
    print(f"\n~) Existing time sequences --------- ")
    from src.time_sequence import _global_abstract_time_sequence
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


def _quick_mesh(*bounds, element_layout=None):
    r"""Make a quick msepy mesh over manifold \Omega = [*bounds].

    This is mainly for test purpose. Thus, we do not need to define abstract objs first.

    Parameters
    ----------
    bounds
    element_layout

    Returns
    -------

    """
    ndim = len(bounds)
    assert 1 <= ndim <= 3, f"bounds must reflect a 1-, 2- or 3-d mesh."
    if element_layout is None:
        if ndim <= 2:
            element_layout = [6 for _ in range(ndim)]
        else:
            element_layout = [3 for _ in range(ndim)]
    else:
        pass
    for i, bound in enumerate(bounds):
        lower, upper = bound
        assert upper > lower, f"bounds[{i}] = {bound} is wrong."
    from src.config import set_embedding_space_dim
    set_embedding_space_dim(ndim)
    from src.manifold import manifold
    manifold = manifold(ndim)
    from src.mesh import mesh
    mesh = mesh(manifold)
    manifold = MsePyManifold(manifold)
    mesh = MsePyMesh(mesh)
    _mf_config(manifold, 'crazy', bounds=bounds)
    _mh_config(mesh, manifold, element_layout)
    return mesh


def find_mesh_of_manifold(msepy_or_abstract_manifold):
    """Find the corresponding msepy mesh."""
    from src.manifold import Manifold
    the_mesh = None

    if msepy_or_abstract_manifold.__class__ is Manifold:
        raise NotImplementedError()

    elif msepy_or_abstract_manifold.__class__ is MsePyManifold:
        for mesh_sym_repr in base['meshes']:
            mesh = base['meshes'][mesh_sym_repr]
            if mesh.manifold is msepy_or_abstract_manifold:
                the_mesh = mesh
                break
    else:
        raise Exception(f"manifold: {msepy_or_abstract_manifold} is not valid")

    assert the_mesh is not None, f"We must have found one!"
    return the_mesh


from msepy.manifold.main import config as _mf_config
from msepy.mesh.main import config as _mh_config


def config(obj):
    """wrapper of _Config class."""
    return _Config(obj)


class _Config(Frozen):
    """"""
    def __init__(self, obj):
        """"""
        self._obj = obj
        self._freeze()

    def __call__(self, *args, **kwargs):                        # Can config the following objects:
        """"""
        if self._obj.__class__ is MsePyManifold:                # 1: config msepy manifold
            return _mf_config(self._obj, *args, **kwargs)

        elif self._obj.__class__ is MsePyMesh:                  # 2: config msepy mesh
            mesh = self._obj
            abstract_mesh = mesh.abstract
            abstract_manifold = abstract_mesh.manifold
            mnf = None
            for mnf_sr in base['manifolds']:
                mnf = base['manifolds'][mnf_sr]
                if mnf.abstract is abstract_manifold:
                    break
            assert mnf is not None, f"cannot find a valid mse-py-manifold."
            return _mh_config(self._obj, mnf, *args, **kwargs)

        else:
            raise NotImplementedError()


# make new stuffs later on ...
def new(abstract_obj):
    """"""
    if abstract_obj._is_space():   #
        ab_sp_sym_repr = abstract_obj._sym_repr
        if ab_sp_sym_repr in base['spaces']:
            space = base['spaces'][ab_sp_sym_repr]
        else:
            space = MsePySpace(abstract_obj)
            base['spaces'][ab_sp_sym_repr] = space
        assert abstract_obj._objective is space, f"Make sure we did not switch implementation!"
        return space
    else:
        raise NotImplementedError()
