# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from typing import Dict
import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msehy.py2.spaces.main import MseHyPy2Space
from msehy.py2.form.cf import MseHyPy2ContinuousForm
from msehy.py2.form.cochain.main import MseHyPy2Cochain


class MseHyPy2RootForm(Frozen):
    """"""

    def __init__(self, abstract_root_form):
        """"""
        self._abstract = abstract_root_form
        abstract_space = abstract_root_form.space
        self._space = abstract_space._objective
        assert self.space.__class__ is MseHyPy2Space, f"space must be a {MseHyPy2Space}."
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
        self._cochain = None
        self._freeze()

    def __repr__(self):
        """repr"""
        ab_rf_repr = self._abstract.__repr__().split(' at ')[0][1:]
        return "<MseHyPy2 " + ab_rf_repr + '>'

    def _is_base(self):
        """Am I a base root-form (not abstracted at a time.)"""
        return self._base is None

    @property
    def _base(self):
        """The base root-form of me."""
        return self._pAti_form['base_form']

    @property
    def abstract(self):
        """the abstract object this root-form is for."""
        return self._abstract

    @property
    def space(self):
        """The `MsePySpace` I am in."""
        return self._space

    @property
    def degree(self):
        """The degree of my space."""
        return self._degree

    @property
    def mesh(self):
        """The objective mesh."""
        return self.space.mesh

    @property
    def cf(self):
        """Continuous form (a shell, the real `cf` is in `cf.field`) of this root-form."""
        if self._cf is None:
            self._cf = MseHyPy2ContinuousForm(self)
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
            self._cochain = MseHyPy2Cochain(self)
        return self._cochain


if __name__ == '__main__':
    # python msehy/py2/form/main.py
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

    msehy, obj = ph.fem.apply('msehy', locals())

    # spaces = msepy.base['spaces']
    # for sym in spaces:
    #     space = spaces[sym]
    #     print(space.mesh)

    manifold = obj['manifold']
    mesh = obj['mesh']

    msehy.config(manifold)('crazy', c=0.0, bounds=[[0, 2], [0, 2]], periodic=False)
    msehy.config(mesh)(15)

    # mesh.visualize()

    # for msh in msehy.base['meshes']:
    #     msh = msehy.base['meshes'][msh]
    #     # msh.visualize()
    #     print(msh.generations[-1])

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

    f0i.cf = scalar
    f0o.cf = scalar
    f2.cf = scalar

    print(f2.space[f2.degree])

    # def refining_strength(x, y):
    #     """"""
    #     return np.sin(2*np.pi*x) * np.cos(2*np.pi*y)
    #
    # # mesh.visualize()
    # mesh.renew(
    #     {0: refining_strength}, [0.3, 0.5, 0.7, 0.9]
    # )

    # f1o.cf = vector
    # f2[0].reduce()
    # f1o[0].reduce()
    # d_f1o = f1o.d()
    # # d_f1o[None].visualize()
    #
    # f = d_f1o - f2
    # f2[None].visualize()
    # f[None].visualize()
