# -*- coding: utf-8 -*-
r"""
"""
import sys
from typing import Dict

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.form.cf import MsePyContinuousForm
from msepy.form.cochain.main import MsePyRootFormCochain
from msepy.form.static import MsePyRootFormStaticCopy
from msepy.form.visualize.main import MsePyRootFormVisualize
from msepy.form.coboundary import MsePyRootFormCoboundary
from msepy.form.matrix import MsePyRootFormMatrix
from msepy.form.boundary_integrate.main import MsePyRootFormBoundaryIntegrate


class MsePyRootForm(Frozen):
    """"""

    def __init__(self, abstract_root_form):
        """"""
        self._abstract = abstract_root_form
        abstract_space = abstract_root_form.space
        self._space = abstract_space._objective
        degree = self.abstract._degree
        assert degree is not None, f"{abstract_root_form} has no degree."
        self._degree = None  # it will be updated to degree below
        self.space.finite.specify_form(self, degree)  # will make new degree if this degree is not saved.
        self._cf = None
        self._cochain = None  # do not initialize cochain here!
        self._pAti_form: Dict = {
            'base_form': None,   # the base form
            'ats': None,   # abstract time sequence
            'ati': None,   # abstract time instant
        }
        self._ats_particular_forms = dict()   # the abstract forms based on this form.
        self._visualize = None
        self._coboundary = None
        self._matrix = None
        self._boundary_integrate = MsePyRootFormBoundaryIntegrate(self)
        self._reconstruct_matrix = None
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
        return self._space.reconstruction_matrix(
            self._degree, *meshgrid_xi_et_sg, element_range=element_range
        )

    def _find_local_dofs_on(self, m, n):
        """find the local dofs numbering on the `n`-face along `m`-direction of element #`element`."""
        return self._space.find.local_dofs(m, n, self._degree)


if __name__ == '__main__':
    # python msepy/form/main.py
    # import numpy as np
    import __init__ as ph

    space_dim = 2
    ph.config.set_embedding_space_dim(space_dim)

    manifold = ph.manifold(space_dim, is_periodic=False)
    mesh = ph.mesh(manifold)
    # L0 = ph.space.new('Lambda', 0)
    # f0 = L0.make_form('f^0', '0-f')
    # L1 = ph.space.new('Lambda', 1)
    # f1 = L1.make_form('f^1', '1-f')
    # L1o = ph.space.new('Lambda', 1, orientation='outer')
    # f1o = L1o.make_form('f^1', '1-f-o')
    # L1i = ph.space.new('Lambda', 1, orientation='inner')
    # f1i = L1i.make_form('h^1', '1-f-i')
    # L2 = ph.space.new('Lambda', 2)
    # f2 = L2.make_form('f^2', '2-f')
    # L3 = ph.space.new('Lambda', 3)
    # f3 = L3.make_form('f^3', '3-f')

    # df0 = ph.exterior_derivative(f0)

    mesh.partition(r'\Gamma1', r'\Gamma2')
    ph.space.finite((3, 3, 3))

    msepy, obj = ph.fem.apply('msepy', locals())

    # print(msepy.base['meshes'])
    # print()
    # print(msepy.base['manifolds'])

    manifold = obj['manifold']
    mesh = obj['mesh']
    Gamma1 = msepy.base['manifolds'][r"\Gamma1"]
    Gamma2 = msepy.base['manifolds'][r"\Gamma2"]
    # print(obj)

    # f0 = obj['f0']
    # f1 = obj['f1']
    #
    # f1o = obj['f1o']
    # f1i = obj['f1i']
    # f2 = obj['f2']
    # f3 = obj['f3']
    #
    # msepy.config(manifold)('crazy', c=0., periodic=False, bounds=[[0, 2] for _ in range(space_dim)])
    msepy.config(manifold)('crazy', c=0.0, bounds=[[0, 2] for _ in range(space_dim)], periodic=False)
    msepy.config(Gamma1)(manifold, {0: [0, 0, 1, 0]})
    # # msepy.config(mnf)('backward_step')
    # msepy.config(mesh)((2, 2, 2))
    msepy.config(mesh)(5)
    # # msepy.config(mesh)(([3, 3, 2], ))
    # # mesh.visualize()

    # for mesh_repr in msepy.base['meshes']:
    #     mesh = msepy.base['meshes'][mesh_repr]
    #     print(mesh_repr)

    # def fx(t, x, y, z):
    #     return np.cos(2*np.pi*x) * np.cos(np.pi*y) * np.cos(np.pi*z) + t

    # def phi_func(t, x, y, z):
    #     """"""
    #     return - np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z) + t * 1

    # def ux(t, x, y):
    #     return np.sin(np.pi*x) * np.cos(np.pi*y) + t*0
    #
    # def uy(t, x, y):
    #     return np.cos(np.pi*x) * np.sin(np.pi*y) + t*0

    # def uz(t, x, y, z):
    #     return np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.sin(np.pi*z) + t

    # scalar = ph.vc.scalar(phi_func)
    # vector = ph.vc.vector(ux, uy)
    #
    # M0 = f0.matrix.mass
    # M1 = f1.matrix.mass
    # M2 = f2.matrix.mass
    # M3 = f3.matrix.mass

    # gm = f3.cochain.gathering_matrix

    # f3.cf = scalar
    # f3[0].reduce()
    # # f3[0].visualize()
    # print(f3[0].error())
    # print(f3[0].norm())
    # df0 = f0[2].coboundary()
    # print(df0[2].error())

    # f1o.cf = vector
    # f1i.cf = vector
    #
    # f1o[2].reduce()
    # f1o[2].visualize(plot_type='quiver', sampling_factor=0.01)

    # # print(f1[2].error())
    #
    # f2.cf = vector
    # f2[2].reduce()
    # f2[2].visualize()
    # # print(f2[2].error())
    #
    # f3.cf = scalar
    # f3[2].reduce()
    # f3[2].visualize()
    # # print(f3[2].error())

    # f0[2].visualize(f1, f2, f3)

    # print()

    # def fx(t, x, y):
    #     return np.sin(2*np.pi*x) * np.sin(np.pi*y) + t
    #
    # scalar = ph.vc.scalar(fx)
    # f0.cf = scalar
    # f0[2].reduce()
    # # f0[2].visualize()
    # f2.cf = scalar
    # f2[2].reduce()
    # # f2[2].visualize()
    #
    # def ux(t, x, y):
    #     return np.sin(2*np.pi*x) * np.cos(2*np.pi*y) + t
    #
    # def uy(t, x, y):
    #     return np.cos(np.pi*x) * np.sin(np.pi*y) + t
    #
    # vector = ph.vc.vector(ux, uy)
    #
    # f1o.cf = vector
    # f1o[2].reduce()
    # # f1o[2].visualize()
    #
    # # mesh.visualize()
    # f1i.cf = vector
    # f1i[2].reduce()
    # # f1i[2].visualize()
    #
    # M = f1o.matrix.mass

    # def fx(t, x):
    #     return np.sin(2*np.pi*x) + t
    # scalar = ph.vc.scalar(fx)
    #
    # # f1.cf = scalar
    # # f1[2].reduce()
    # # f1[2].visualize()
    #
    # f0.cf = scalar
    # f0[2].reduce()
    # # f0[2].visualize()
    # f_error = f0[2].error()  # by default, we will compute the L^2 error.
    # print(f_error)
    #
    # df0 = f0.coboundary[2]()
    # df_error = df0[2].error()
    # print(f_error, df_error)
    #
    # # df0[2].visualize()
    # # # df0 = f0[2].coboundary()
    # # # print(df0)
