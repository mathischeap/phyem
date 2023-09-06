# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
import sys
from typing import Dict
from random import random
from time import time

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.space.main import MsePySpace
from msepy.form.cf import MsePyContinuousForm
from msepy.form.cochain.main import MsePyRootFormCochain
from msepy.form.cochain.passive import MsePyRootFormCochainPassive
from msepy.form.static import MsePyRootFormStaticCopy
from msepy.form.visualize.main import MsePyRootFormVisualize
from msepy.form.coboundary import MsePyRootFormCoboundary
from msepy.form.matrix import MsePyRootFormMatrix
from msepy.form.boundary_integrate.main import MsePyRootFormBoundaryIntegrate

from tools.miscellaneous.ndarray_cache import ndarray_key_comparer, add_to_ndarray_cache


class MsePyRootForm(Frozen):
    """"""

    def __init__(self, abstract_root_form):
        """"""
        self._abstract = abstract_root_form
        abstract_space = abstract_root_form.space
        self._space = abstract_space._objective
        assert self.space.__class__ is MsePySpace, f"space must be a {MsePySpace}."
        degree = self.abstract._degree
        assert degree is not None, f"{abstract_root_form} has no degree."
        self._degree = None  # it will be updated to degree below
        self.space.finite.specify_form(self, degree)  # will make new degree if this degree is not saved.
        self._pAti_form: Dict = {
            'base_form': None,   # the base form
            'ats': None,   # abstract time sequence
            'ati': None,   # abstract time instant
        }
        self._ats_particular_forms = dict()   # the abstract forms based on this form.
        self._cf = None
        self._cochain = None  # do not initialize cochain here!
        self._visualize = None
        self._coboundary = None
        self._matrix = None
        self._boundary_integrate = MsePyRootFormBoundaryIntegrate(self)
        self._reconstruct_matrix = None
        self._reconstruct_matrix_cache = dict()
        self._numeric = None
        self._freeze()

    def _saving_check(self):
        """If you want to use `ph.save` to save instances of this class, it must have this method."""
        saving_key = self.abstract._pure_lin_repr  # the key used to represent self in the dict to be saved.
        saving_obj = self  # self only presents in one thread, so just save it.
        return saving_key, saving_obj

    @property
    def abstract(self):
        """the abstract object this root-form is for."""
        return self._abstract

    def __repr__(self):
        """repr"""
        ab_rf_repr = self._abstract.__repr__().split(' at ')[0][1:]
        return "<MsePy " + ab_rf_repr + super().__repr__().split(" object")[1]

    def __getitem__(self, t):
        """return the realtime copy of `self` at time `t`."""
        if t is None:
            t = self.cochain.newest  # newest time
        else:
            pass
        t = self.cochain._parse_t(t)  # round off the truncation error to make it clear.
        if isinstance(t, (int, float)):
            if self._is_base():
                return MsePyRootFormStaticCopy(self, t)
            else:
                return MsePyRootFormStaticCopy(self._base, t)
        else:
            raise Exception(f"cannot accept t={t}.")

    def __call__(self, ati=None, **kwargs):
        """We first get a ``time`` from ati(*args, **kwargs), then return a static copy of self at ``time``.

        Parameters
        ----------
        ati
        kwargs

        Returns
        -------

        """
        if self._is_base():
            assert ati is not None, \
                f'For base root form,  provide the abstract time instant, i.e. kwarg: `ati`.'

        else:
            if ati is None:
                ati = self._pAti_form['ati']
            else:
                pass
        t = ati(**kwargs)()
        return self[t]

    def _is_base(self):
        """Am I a base root-form (not abstracted at a time.)"""
        return self._base is None

    @property
    def _base(self):
        """The base root-form I have."""
        return self._pAti_form['base_form']

    @property
    def name(self):
        """name of this form is the pure linguistic representation."""
        return self._abstract._pure_lin_repr

    @property
    def m(self):
        return self.space.m  # esd

    @property
    def n(self):
        return self.space.n  # mesh.ndim

    @property
    def space(self):
        """The `MsePySpace` I am in."""
        return self._space

    @property
    def mesh(self):
        """The objective mesh."""
        return self.space.mesh

    @property
    def degree(self):
        """The degree of my space."""
        return self._degree

    @property
    def cf(self):
        """Continuous form (a shell, the real `cf` is in `cf.field`) of this root-form."""
        if self._cf is None:
            self._cf = MsePyContinuousForm(self)
        return self._cf

    @cf.setter
    def cf(self, _cf):
        """Setter of `cf`.

        We actually set `cf.field`, use a shell `cf` to enabling extra checkers and so on.
        """
        self.cf.field = _cf

    def set_cf(self, cf):
        """A more reasonable method name."""
        self.cf = cf

    def d(self):
        """d(self)"""
        E = self.coboundary.incidence_matrix._data.toarray()

        def _d_cochain(t):
            self_cochain_t = self.cochain[t].local
            d_cochain_at_t = np.einsum(
                'ij, kj -> ki',
                E, self_cochain_t,
                optimize='optimal',
            )
            return d_cochain_at_t

        df = self.coboundary._make_df()
        df._cochain = MsePyRootFormCochainPassive(df, reference_form_cochain=self.cochain)
        df._cochain._realtime_local_cochain_caller = _d_cochain

        cf = self.cf.field

        if cf is None:
            pass
        else:
            # cannot use cf.field.exterior_derivative().
            df.cf = self.cf.exterior_derivative()

        return df

    def _copy(self):
        """Make a copy of df of empty cochain; do not specify cochain."""
        ab_space = self.abstract.space
        sym_repr = str(hash(random() + time()))      # random sym_repr <-- important, do not try to print its repr
        lin_repr = str(hash(random() + time() + 2))  # random lin_repr <-- important, do not try to print its repr
        # The below abstract root-form is not recorded.
        ab_f = self.abstract.__class__(ab_space, sym_repr, lin_repr, True, update_cache=False)
        ab_f.degree = self.degree
        f = self.__class__(ab_f)
        return f

    def __sub__(self, other):
        """self - other"""
        if other.__class__ is self.__class__:

            assert other.mesh == self.mesh, f"meshes do not match"
            assert other.space == self.space, f"spaces do not match"
            assert other.degree == self.degree, f"degrees do not match"

            def cochain_sub_caller(t):
                self_cochain_t = self.cochain[t].local
                other_cochain_t = other.cochain[t].local
                return self_cochain_t - other_cochain_t

            f = self._copy()
            f._cochain = MsePyRootFormCochainPassive(
                f,
                reference_form_cochain=[self.cochain, other.cochain]
            )
            f._cochain._realtime_local_cochain_caller = cochain_sub_caller

            scf = self.cf.field
            ocf = other.cf.field

            if scf is None or ocf is None:
                pass
            else:
                f.cf = scf - ocf

            return f

        else:
            raise NotImplementedError(f"{other}")

    def __add__(self, other):
        """self - other"""
        if other.__class__ is self.__class__:

            assert other.mesh == self.mesh, f"meshes do not match"
            assert other.space == self.space, f"spaces do not match"
            assert other.degree == self.degree, f"degrees do not match"

            def cochain_add_caller(t):
                self_cochain_t = self.cochain[t].local
                other_cochain_t = other.cochain[t].local
                return self_cochain_t + other_cochain_t

            f = self._copy()
            f._cochain = MsePyRootFormCochainPassive(
                f,
                reference_form_cochain=[self.cochain, other.cochain]
            )
            f._cochain._realtime_local_cochain_caller = cochain_add_caller

            scf = self.cf.field
            ocf = other.cf.field

            if scf is None or ocf is None:
                pass
            else:
                f.cf = scf + ocf

            return f

        else:
            raise NotImplementedError(f"{other}")

    @property
    def cochain(self):
        """The cochain class."""
        if self._cochain is None:
            self._cochain = MsePyRootFormCochain(self)
        return self._cochain

    def reduce(self, t, update_cochain=True, target=None, **kwargs):
        """reduce `self.cf` if ``targe`` is None else ``target``
        at time `t` and decide whether update the cochain.
        """
        if target is None:
            cochain_local = self.space.reduce(self.cf, t, self.degree, **kwargs)

        else:
            # remember, space reduce only accept cf object. So we do the following
            if target.__class__ is MsePyContinuousForm:
                pass
            else:
                template_cf = MsePyContinuousForm(self)  # make a new `cf`, it does not affect the `cf` of self.
                template_cf.field = target
                target = template_cf

            cochain_local = self.space.reduce(target, t, self.degree, **kwargs)

        if update_cochain:
            self[t].cochain = cochain_local
        else:
            pass

        return cochain_local

    def reconstruct(self, t, *meshgrid, **kwargs):
        """Reconstruct self at time `t`."""
        if t is None:
            t = self.cochain.newest
        else:
            assert isinstance(t, (int, float)), f"t={t} type wrong!"
        local_cochain = self.cochain[t].local
        degree = self.degree
        return self.space.reconstruct(local_cochain, degree, *meshgrid, **kwargs)

    def _evaluate_bf_on(self, *meshgrid):
        """Evaluate the basis functions of this form (the space)."""
        return self._space.basis_functions[self.degree](*meshgrid)

    @property
    def visualize(self):
        """visualize"""
        if self._visualize is None:
            self._visualize = MsePyRootFormVisualize(self)
        return self._visualize

    def error(self, t=None, quad_degree=None, **kwargs):
        """error"""
        if t is None:
            t = self.cochain.newest
        else:
            assert isinstance(t, (int, float)), f"t={t} type wrong!"
        local_cochain = self.cochain[t].local
        degree = self.degree
        return self.space.error(self.cf, t, local_cochain, degree, quad_degree=quad_degree, **kwargs)

    def norm(self, t=None, quad_degree=None, **kwargs):
        """norm"""
        if t is None:
            t = self.cochain.newest
        else:
            assert isinstance(t, (int, float)), f"t={t} type wrong!"
        local_cochain = self.cochain[t].local
        degree = self.degree
        return self.space.norm(local_cochain, degree, quad_degree=quad_degree, **kwargs)

    @property
    def coboundary(self):
        """coboundary"""
        if self._coboundary is None:
            self._coboundary = MsePyRootFormCoboundary(self)
        return self._coboundary

    @property
    def matrix(self):
        """matrix"""
        if self._matrix is None:
            self._matrix = MsePyRootFormMatrix(self)
        return self._matrix

    @property
    def boundary_integrate(self):
        return self._boundary_integrate

    def reconstruction_matrix(self, *meshgrid_xi_et_sg, element_range=None):
        """compute reconstruction matrices for particular elements."""
        cached, data = ndarray_key_comparer(
            self._reconstruct_matrix_cache, meshgrid_xi_et_sg, check_str=str(element_range)
        )
        if cached:
            return data
        else:
            pass

        data = self._space.reconstruction_matrix(
            self._degree, *meshgrid_xi_et_sg, element_range=element_range
        )
        add_to_ndarray_cache(
            self._reconstruct_matrix_cache, meshgrid_xi_et_sg, data, check_str=str(element_range)
        )
        return data

    def _find_local_dofs_on(self, m, n):
        """find the local dofs numbering on the `n`-face along `m`-direction of element #`element`."""
        return self._space.find.local_dofs(m, n, self._degree)


if __name__ == '__main__':
    # python msepy/form/main.py
    import __init__ as ph

    space_dim = 2
    ph.config.set_embedding_space_dim(space_dim)

    manifold = ph.manifold(space_dim, is_periodic=False)
    mesh = ph.mesh(manifold)

    L0i = ph.space.new('Lambda', 0, orientation='inner')
    L0o = ph.space.new('Lambda', 0, orientation='outer')
    L1i = ph.space.new('Lambda', 1, orientation='inner')
    L1o = ph.space.new('Lambda', 1, orientation='outer')
    L2 = ph.space.new('Lambda', 2)

    f0i = L0i.make_form('f_i^0', '0-f-i')
    f0o = L0o.make_form('f_o^0', '0-f-o')
    f1i = L1i.make_form('f_i^1', '1-f-i')
    f1o = L1o.make_form('f_o^1', '1-f-o')
    f2 = L2.make_form('f^2', '2-f')

    ph.space.finite(5)

    msepy, obj = ph.fem.apply('msepy', locals())

    manifold = obj['manifold']
    mesh = obj['mesh']

    msepy.config(manifold)('crazy', c=0.0, bounds=[[0, 2] for _ in range(space_dim)], periodic=False)
    msepy.config(mesh)(15)

    f0i = obj['f0i']
    f0o = obj['f0o']
    f1i = obj['f1i']
    f1o = obj['f1o']
    f2 = obj['f2']


    def fx(t, x, y):
        return np.sin(2*np.pi*x) * np.sin(np.pi*y) + t


    def ux(t, x, y):
        return np.sin(2*np.pi*x) * np.cos(2*np.pi*y) + t


    def uy(t, x, y):
        return -np.cos(2*np.pi*x) * np.sin(2*np.pi*y) + t


    scalar = ph.vc.scalar(fx)
    vector = ph.vc.vector(ux, uy)

    f1o.cf = vector
    f2.cf = scalar

    f2[0].reduce()
    f1o[0].reduce()
    d_f1o = f1o.d()
    # d_f1o[None].visualize()

    f = d_f1o - f2
    f2[None].visualize()
    f[None].visualize()
