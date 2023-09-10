# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from typing import Dict
import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msehy.py2.space.main import MseHyPy2Space
from msehy.py2.form.cf import MseHyPy2ContinuousForm
from msehy.py2.form.cochain.main import MseHyPy2Cochain
from msehy.py2.form.static import MseHyPy2RootFormStaticCopy
from msehy.py2.form.visualize.main import MseHyPy2RootFormVisualize
from msehy.py2.form.coboundary import MseHyPy2RootFormCoboundary


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
        self._visualize = None
        self._coboundary = None
        self._freeze()

    def __repr__(self):
        """repr"""
        ab_rf_repr = self._abstract.__repr__().split(' at ')[0][1:]
        return "<MseHyPy2 " + ab_rf_repr + '>'

    def __getitem__(self, t_g):
        """return the realtime copy of `self` at time `t`."""
        if t_g is None:
            t_g = self.cochain.newest  # newest time
        else:
            pass
        t, g = t_g
        t = self.cochain._parse_t(t)  # round off the truncation error to make it clear.
        g = self._pg(g)
        if isinstance(t, (int, float)):
            if self._is_base():
                return MseHyPy2RootFormStaticCopy(self, t, g)
            else:
                return MseHyPy2RootFormStaticCopy(self._base, t, g)
        else:
            raise Exception(f"cannot accept t={t}.")

    def _is_base(self):
        """Am I a base root-form (not abstracted at a time.)"""
        return self._base is None

    @property
    def _base(self):
        """The base root-form of me."""
        return self._pAti_form['base_form']

    @property
    def name(self):
        """name of this form is the pure linguistic representation."""
        return self._abstract._pure_lin_repr

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

    def _pt(self, t):
        """parse t"""
        return self.cochain._parse_t(t)

    def _pg(self, generation):
        """parse generation."""
        return self.mesh._pg(generation)

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

    def reduce(self, t, g, update_cochain=True, target=None, **kwargs):
        """reduce `self.cf` if ``targe`` is None else ``target``
        at time `t`, on generation `g`, and decide whether update the cochain.
        """
        g = self._pg(g)

        if target is None:
            cochain_local = self.space.reduce(self.cf, t, g, self.degree, **kwargs)

        else:
            # remember, space reduce only accept cf object. So we do the following
            if target.__class__ is MseHyPy2ContinuousForm:
                pass
            else:
                template_cf = MseHyPy2ContinuousForm(self)
                # make a new `cf`, it does not affect the `cf` of self.
                template_cf.field = target
                target = template_cf

            cochain_local = self.space.reduce(target, t, g, self.degree, **kwargs)

        if update_cochain:
            self[(t, g)].cochain = cochain_local
        else:
            pass

        return cochain_local

    def reconstruct(self, t, g, *meshgrid, **kwargs):
        """Reconstruct self at time `t`."""
        if t is None:
            t = self.cochain.newest
        else:
            assert isinstance(t, (int, float)), f"t={t} type wrong!"

        cochain = self.cochain[(t, g)]
        assert self.degree == cochain._f.degree, f"degree does not match"
        return self.space.reconstruct(g, cochain, *meshgrid, **kwargs)

    def error(self, t=None, g=None, quad_degree=None, **kwargs):
        """error"""
        if t is None:
            t = self.cochain.newest[0]
        else:
            assert isinstance(t, (int, float)), f"t={t} type wrong!"

        if g is None:
            g = self.cochain.newest[1]
        else:
            pass
        g = self._pg(g)
        cochain = self.cochain[(t, g)]
        assert self.degree == cochain._f.degree, f"degree does not match"
        return self.space.error(self.cf, cochain, quad_degree=quad_degree, **kwargs)

    def norm(self, t=None, g=None, quad_degree=None, **kwargs):
        """norm."""
        if t is None:
            t = self.cochain.newest[0]
        else:
            assert isinstance(t, (int, float)), f"t={t} type wrong!"

        if g is None:
            g = self.cochain.newest[1]
        else:
            pass
        g = self._pg(g)
        cochain = self.cochain[(t, g)]
        assert self.degree == cochain._f.degree, f"degree does not match"
        return self.space.norm(cochain, quad_degree=quad_degree, **kwargs)

    @property
    def visualize(self):
        """visualize."""
        if self._visualize is None:
            self._visualize = MseHyPy2RootFormVisualize(self)
        return self._visualize

    @property
    def coboundary(self):
        """coboundary."""
        if self._coboundary is None:
            self._coboundary = MseHyPy2RootFormCoboundary(self)
        return self._coboundary


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

    ph.space.finite(4)

    msehy, obj = ph.fem.apply('msehy', locals())

    # spaces = msepy.base['spaces']
    # for sym in spaces:
    #     space = spaces[sym]
    #     print(space.mesh)

    manifold = obj['manifold']
    mesh = obj['mesh']

    msehy.config(manifold)('crazy', c=0.0, bounds=[[0, 2], [0, 2]], periodic=False)
    msehy.config(mesh)(7)

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
    f1i.cf = vector
    f1o.cf = vector
    f2.cf = scalar

    def refining_strength(x, y):
        """"""
        return np.sin(2*np.pi*x) * np.cos(2*np.pi*y)

    mesh.renew(
        {0: refining_strength}, [0.3, 0.5]
    )
    # mesh.visualize()

    # _ = mesh.current_representative.opposite_pairs
    # f1i.cochain_switch_matrix()
    # f1o.cochain_switch_matrix()

    f0i[(0, 1)].reduce()
    f0o[(0, 1)].reduce()
    f1i[(0, 1)].reduce()
    f1o[(0, 1)].reduce()
    f2[(0, 1)].reduce()

    print(f0i[(0, 1)].error())
    print(f0o[(0, 1)].error())
    print(f1i[(0, 1)].error())
    print(f1o[(0, 1)].error())
    print(f2[(0, 1)].error())
    #
    # f0i[(0, 1)].visualize(saveto='f0i.vtk')
    # f0o[(0, 1)].visualize(saveto='f0o.vtk')
    # f1i[(0, 1)].visualize(saveto='f1i.vtk')
    # f1o[(0, 1)].visualize(saveto='f1o.vtk')
    # f2[(0, 1)].visualize(saveto='f2.vtk')

    # gm = f0i.cochain.gathering_matrix(1)
    # gm = f0o.cochain.gathering_matrix(1)
    # gm = f1i.cochain.gathering_matrix(1)
    # gm = f1o.cochain.gathering_matrix(1)
    # gm = f2.cochain.gathering_matrix(1)

    # rc = f1i.space.local_dof_representative_coordinates(f0i.degree)

    # print(rc)

    # for i in gm:
    #     print(i, gm[i])

    # print(f2.space[f2.degree])
    # _ = f2.cochain.local_numbering
    # f2_0 = f2.space[1]
    # _ = f2.space.basis_functions
    # print(f2.space[3].gathering_matrix)

    # f0i.space.basis_functions[3](np.array([-1,0,1]), np.array([-1,0,1]))

    # # f2.cochain.gathering_matrix(-1)
    # mesh.renew(
    #     {0: refining_strength}, [0.3, 0.5, 0.7, 0.9]
    # )
    # f2.cochain.gathering_matrix(-1)

    # mesh.renew(
    #     {0: refining_strength}, [0.3, 0.5, 0.7, 0.9]
    # )
    # f2.cochain.gathering_matrix(-1)

    # f1o.cf = vector
    # f2[0].reduce()
    # f1o[0].reduce()
    # d_f1o = f1o.d()
    # # d_f1o[None].visualize()
    #
    # f = d_f1o - f2
    # f2[None].visualize()
    # f[None].visualize()
