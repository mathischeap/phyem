# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 1:22 PM on 4/17/2023
"""

import sys
from typing import Dict

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.form.cf import MsePyContinuousForm
from msepy.form.cochain.main import MsePyRootFormCochain
from msepy.form.realtime import MsePyRootFormRealTimeCopy
from msepy.form.visualize.main import MsePyRootFormVisualize
from msepy.form.error.main import MsePyRootFormError
from msepy.form.coboundary import MsePyRootFormCoboundary
from msepy.form.matrix import MsePyRootFormMatrix


class MsePyRootForm(Frozen):
    """"""

    def __init__(self, abstract_root_form):
        """"""
        self._abstract = abstract_root_form
        abstract_space = abstract_root_form.space
        self._space = abstract_space._objective
        degree = self.abstract._degree
        assert degree is not None, f"{abstract_root_form} has no degree."
        self._degree = None
        self.space.finite.specify_form(self, degree)  # will make new degree if this degree is not saved.
        self._cf = None
        self._cochain = None  # do not initialize cochain here!
        self._pAti_form: Dict = {
            'base_form': None,
            'ats': None,
            'ati': None
        }
        self._ats_particular_forms = dict()   # the abstract forms based on this form.
        self._numbering = None
        self._reduce = None
        self._reconstruct = None
        self._visualize = None
        self._error = None
        self._coboundary = None
        self._matrix = None
        self._freeze()

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
        if self._is_base():
            return MsePyRootFormRealTimeCopy(self, t)
        else:
            return MsePyRootFormRealTimeCopy(self._base, t)

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
        """Continuous form (a shell, the real `cf` is in `cf.field`) of this root-form"""
        if self._cf is None:
            self._cf = MsePyContinuousForm(self)
        return self._cf

    @cf.setter
    def cf(self, cf):
        """Setter of `cf`.

        We actually set `cf.field`, use a shell `cf` to enabling extra checkers and so on.
        """
        self.cf.field = cf

    def set_cf(self, cf):
        """A more reasonable method name."""
        self.cf = cf

    @property
    def cochain(self):
        """The cochain class."""
        if self._cochain is None:
            self._cochain = MsePyRootFormCochain(self)
        return self._cochain

    def reduce(self, t, update_cochain=True, **kwargs):
        """reduce `self.cf` at time `t` and decide whether update the cochain."""
        cochain_local = self.space.reduce(self.cf, t, self.degree, **kwargs)
        if update_cochain:
            self[t].cochain = cochain_local
        else:
            pass
        return cochain_local

    def reconstruct(self, t, *meshgrid, **kwargs):
        """Reconstruct self at time `t`."""
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

    @property
    def error(self):
        """visualize"""
        if self._error is None:
            self._error = MsePyRootFormError(self)
        return self._error

    @property
    def coboundary(self):
        """visualize"""
        if self._coboundary is None:
            self._coboundary = MsePyRootFormCoboundary(self)
        return self._coboundary

    @property
    def matrix(self):
        """visualize"""
        if self._matrix is None:
            self._matrix = MsePyRootFormMatrix(self)
        return self._matrix


if __name__ == '__main__':
    # python msepy/form/main.py
    import numpy as np
    import __init__ as ph

    space_dim = 3
    ph.config.set_embedding_space_dim(space_dim)

    manifold = ph.manifold(space_dim)
    mesh = ph.mesh(manifold)
    L0 = ph.space.new('Lambda', 0)
    f0 = L0.make_form('f^0', '0-f')
    L1 = ph.space.new('Lambda', 1)
    f1 = L1.make_form('f^1', '1-f')
    # L1o = ph.space.new('Lambda', 1, orientation='outer')
    # f1o = L1o.make_form('f^1', '1-f-o')
    # L1i = ph.space.new('Lambda', 1, orientation='inner')
    # f1i = L1i.make_form('h^1', '1-f-i')
    L2 = ph.space.new('Lambda', 2)
    f2 = L2.make_form('f^2', '2-f')
    L3 = ph.space.new('Lambda', 3)
    f3 = L3.make_form('f^3', '3-f')

    df0 = ph.exterior_derivative(f0)

    ph.space.finite((3, 4, 5))

    msepy, obj = ph.fem.apply('msepy', locals())

    manifold = obj['manifold']
    mesh = obj['mesh']
    print(obj)

    # f0 = obj['f0']
    # f1 = obj['f1']
    #
    # # f1o = obj['f1o']
    # # f1i = obj['f1i']
    # f2 = obj['f2']
    # f3 = obj['f3']
    #
    # # msepy.config(manifold)('crazy', c=0., periodic=False, bounds=[[0, 2] for _ in range(space_dim)])
    # msepy.config(manifold)('crazy_multi', c=0.0, bounds=[[0, 2] for _ in range(space_dim)])
    # # msepy.config(mnf)('backward_step')
    # msepy.config(mesh)(([3, 3, 3, 3, 3], [1, 1, 1, 1, 1], [2, 2, 3, 3]))
    # # msepy.config(mesh)(10)
    # # msepy.config(mesh)(([3, 3, 2], ))
    # # mesh.visualize()
    #
    # def fx(t, x, y, z):
    #     return np.cos(2*np.pi*x) * np.cos(np.pi*y) * np.cos(np.pi*z) + t
    #
    # def ux(t, x, y, z):
    #     return np.sin(np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z) + t
    #
    # def uy(t, x, y, z):
    #     return np.cos(2*np.pi*x) * np.sin(np.pi*y) * np.cos(2*np.pi*z) + t
    #
    # def uz(t, x, y, z):
    #     return np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.sin(np.pi*z) + t
    #
    # scalar = ph.vc.scalar(fx)
    # vector = ph.vc.vector(ux, uy, uz)
    #
    # M0 = f0.matrix.mass
    # M1 = f1.matrix.mass
    # M2 = f2.matrix.mass
    # M3 = f3.matrix.mass

    # gm = f3.cochain.gathering_matrix

    # f0.cf = scalar
    # f0[2].reduce()
    # f0[2].visualize()
    # print(f0[2].error())
    # df0 = f0[2].coboundary()
    # print(df0[2].error())

    # f1.cf = vector
    # f1[2].reduce()
    # f1[2].visualize()
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
