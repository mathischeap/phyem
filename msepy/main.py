# -*- coding: utf-8 -*-
r"""

.. note::

    *msepy* stands for **mimetic spectral elements in Python**. It is an implementation of
    the mimetic spectral element methods using pure Python. In other words, the most
    computationally intensive part of the simulation, the linear system solving, is also
    done within Python.

    **Advantages** of this are clear. No external packages (or APIs to other kernels)
    are needed to use *msepy* implementation; only common Python packages like scipy, numpy,
    matplotlib, etc., are required. Users can quickly settle their machines no mather they are
    Windows, Linux or Mac.

    The most obvious **disadvantage** of this implementation is that, since Python suffers from
    its relatively lower speed, this implementation is not proper for large problems. It is
    hard to say what problems are large. And it is also dependent on the machine. Normally,
    *msepy* is very handy for 1- or 2-dimentional and small 3-dimensional problems. User could
    explore the edge by trial.

To invoke the msepy implementation, use indicator ``'msepy'`` as the
first argument for ``apply`` of ``fem`` module, i.e.,

>>> implementation, objects = ph.fem.apply('msepy', locals())

To pick up the implemented counterpart of an abstract instance, we can just use its variable name, for example,

>>> manifold = objects['manifold']
>>> mesh = objects['mesh']

If some instances have no explicit varialbes, you could possiblly
pick them up using their symbolic representations.
For example, the boundary manifolds for defineing the boundary conditions have no explicit varibles,
we can pick them using theire symbolic representations through the dictionary ``implementation.base``,

>>> Gamma_alpha = implementation.base['manifolds'][r"\Gamma_{\alpha}"]
>>> Gamma_beta = implementation.base['manifolds'][r"\Gamma_{\beta}"]

For these instances that can be accessed throug ``implementation.base``,
there are just four types, manifolds, meshes, spaces and forms. They can be accessed
with keys ``'manifolds'``, ``'meshes'``, ``'spaces'`` and ``'forms'``,
respectively. For example

>>> mesh is implementation.base['meshes'][r'\mathfrak{M}']
True


.. _Implementations-msepy-config:

Configuration
*************

>>> implementation.config(manifold)(
...     'crazy', c=0., bounds=([0, 1], [0, 1]), periodic=False,
... )




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
