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
from msehy.py2.form.cochain.passive import MseHyPy2RootFormCochainPassive

from tools.miscellaneous.ndarray_cache import add_to_ndarray_cache, ndarray_key_comparer


_reconstruct_matrix_cache = {}


class MseHyPy2RootForm(Frozen):
    """"""

    def __init__(self, abstract_root_form):
        """"""
        self._abstract = abstract_root_form
        abstract_space = abstract_root_form.space
        self._space = abstract_space._objective
        assert self.space.__class__ is MseHyPy2Space, f"space {self.space} must be a {MseHyPy2Space}."
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
        self._numeric = None
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

    @staticmethod
    def _is_discrete_form():
        """A signature."""
        return True

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

    def d(self):
        """d(self)"""

        def _d_cochain(t_g):
            t, g = t_g
            t = self.cochain._parse_t(t)
            g = self._pg(g)
            E = self.space.incidence_matrix(self.degree, g)
            self_cochain_tg = self.cochain[(t, g)].local
            d_cochain_at_tg = dict()
            for index in self_cochain_tg:
                d_cochain_at_tg[index] = E[index] @ self_cochain_tg[index]
            return d_cochain_at_tg

        df = self.coboundary._make_df()
        df._cochain = MseHyPy2RootFormCochainPassive(df, reference_form_cochain=self.cochain)
        df._cochain._realtime_local_cochain_caller = _d_cochain

        cf = self.cf.field

        if cf is None:
            pass
        else:
            # cannot use cf.field.exterior_derivative().
            df.cf = self.cf.exterior_derivative()

        return df

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

    def reconstruction_matrix(self, *meshgrid_xi_et, g, fc_range=None):
        """compute reconstruction matrices for particular elements."""
        if self._is_base():
            _form_id = id(self)
        else:
            _form_id = id(self._pAti_form['base_form'])

        g = self._pg(g)
        check_str = str(_form_id) + '-' + str(g) + str(fc_range)

        cached, data = ndarray_key_comparer(
            _reconstruct_matrix_cache, meshgrid_xi_et, check_str=check_str
        )
        if cached:
            return data
        else:
            pass

        data = self._space.reconstruction_matrix(
            self._degree, g, *meshgrid_xi_et, fc_range=fc_range
        )

        add_to_ndarray_cache(
            _reconstruct_matrix_cache, meshgrid_xi_et, data, check_str=check_str
        )

        return data

    def evolve(self, source_t_g=None, des_generation=None):
        """We evolve the form using its cochain (source_t_g) (on the corresponding generation) to the
        `target_generation`. The new cochain is saved to cochain key (t, target_generation).

        if `source_t_g` (`t, g = source_t_g`) is None, we use the newest time and newest generation.

        If `target_generation` is None, we use `target_generation = self.mesh._pg(-1)`, i.e., the
        most recent generation.

        After parsing t, g, target_generation, if we found g == target_generation, raise Error.
        """
        if source_t_g is None:
            source_t_g = self.cochain.newest
        else:
            pass
        t, g = source_t_g
        t = self.cochain._parse_t(t)
        g = self._pg(g)
        assert (t, g) in self.cochain, f"cochain @ {(t, g)} is not available."
        dg = self._pg(des_generation)
        assert g != dg, f"Source generation {g} must be different from Destination generation {dg}."
        assert g in self.mesh, f"Source generation {g} is not available."
        assert dg in self.mesh, f"Destination generation {dg} is not available."

        link = self.mesh.link(dg, g)

        indicator = self.space.abstract.indicator

        _Lambda_1 = False
        if indicator == 'Lambda':
            k = self.space.abstract.k
            if k == 1:
                _Lambda_1 = True

        old_csm = self._space.basis_functions.cochain_switch_matrix(self.degree, g)
        num_dofs_components = self._space.num_local_dof_components(self.degree)

        sour_cochain = self.cochain[(t, g)].local
        dest_cochain: Dict = dict()
        for dest_index in link:
            source_indices = link[dest_index]
            if source_indices is None:   # cell is the same, cannot just pass it to the destination.
                _ = sour_cochain[dest_index]
                if _Lambda_1 and dest_index in old_csm:
                    csm0, csm1 = old_csm[dest_index]
                    if isinstance(dest_index, str):
                        num_components = num_dofs_components['t'][0]
                    else:
                        num_components = num_dofs_components['q'][0]

                    dest_cochain[dest_index] = np.concatenate([
                        csm0 @ _[:num_components],
                        csm1 @ _[num_components:]
                    ])
                else:
                    dest_cochain[dest_index] = _

            else:
                if isinstance(source_indices, list):
                    # dest cell is coarser: get cochain from multiple smaller cell.
                    assert len(source_indices) > 1, f'Must be!'
                    local_source_cochain = dict()
                    for si in source_indices:
                        local_source_cochain[si] = sour_cochain[si]
                    cochain = self.space.coarsen(
                        self.degree, (dg, dest_index), (g, source_indices, local_source_cochain)
                    )
                else:  # dest cell is more refined: get cochain from a bigger cell.
                    cochain = self.space.refine(
                        self.degree, (dg, dest_index), (g, source_indices, sour_cochain[source_indices])
                    )
                dest_cochain[dest_index] = cochain

        if _Lambda_1:
            csm = self._space.basis_functions.cochain_switch_matrix(self.degree, dg)
            for e in csm:
                csm0, csm1 = csm[e]
                _ = dest_cochain[e]
                if isinstance(e, str):
                    num_components = num_dofs_components['t'][0]
                else:
                    num_components = num_dofs_components['q'][0]

                dest_cochain[e] = np.concatenate([
                    csm0 @ _[:num_components],
                    csm1 @ _[num_components:]
                ])
        else:
            pass

        self.cochain._set(t, dg, dest_cochain)


if __name__ == '__main__':
    # python msehy/py2/form/main.py
    import __init__ as ph

    space_dim = 2
    ph.config.set_embedding_space_dim(space_dim)

    # manifold = ph.manifold(space_dim, is_periodic=True)
    manifold = ph.manifold(space_dim, is_periodic=False)
    mesh = ph.mesh(manifold)

    L0i = ph.space.new('Lambda', 0, orientation='inner')
    L0o = ph.space.new('Lambda', 0, orientation='outer')
    L1i = ph.space.new('Lambda', 1, orientation='inner')
    L1o = ph.space.new('Lambda', 1, orientation='outer')
    L2o = ph.space.new('Lambda', 2, orientation='outer')
    L2i = ph.space.new('Lambda', 2, orientation='inner')

    f0i = L0i.make_form('f_i^0', '0-f-i')
    f0o = L0o.make_form('f_o^0', '0-f-o')
    f1i = L1i.make_form('f_i^1', '1-f-i')
    f1o = L1o.make_form('f_o^1', '1-f-o')
    f2o = L2o.make_form('f^2', '2-f')

    ph.space.finite(3)

    msehy, obj = ph.fem.apply('msehy', locals())

    # spaces = msepy.base['spaces']
    # for sym in spaces:
    #     space = spaces[sym]
    #     print(space.mesh)

    manifold = obj['manifold']
    mesh = obj['mesh']

    # msehy.config(manifold)('crazy', c=0., bounds=([-1, 1], [-1, 1]), periodic=True)
    msehy.config(manifold)('crazy', c=0.0, bounds=[[0, 1], [0, 1]], periodic=False)

    msehy.config(mesh)(13)

    # mesh.visualize()

    # for msh in msehy.base['meshes']:
    #     msh = msehy.base['meshes'][msh]
    #     # msh.visualize()
    #     print(msh.generations[-1])

    f0i = obj['f0i']
    f0o = obj['f0o']
    f1i = obj['f1i']
    f1o = obj['f1o']
    f2o = obj['f2o']


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
    f2o.cf = scalar
    # f1o[(0, 0)].reduce()
    # f1o[(0, 0)].visualize.quick()

    from msehy.py2.tools.random_refining_strength_function import RandomRefiningStrengthFunction
    random_refining_strength = RandomRefiningStrengthFunction([[0, 1], [0, 1]])
    mesh.renew(
        random_refining_strength, [0.3, 0.5]
    )
    rep = mesh.current_representative.map
    for i in rep:
        print(i, rep[i])

    mesh.visualize()
    # f1o[(0, 1)].reduce()
    # M = f2o.space.mass_matrix(3, 1)

    # f1o.evolve()
    # f1o[(0, 1)].visualize.quick()

    random_refining_strength = RandomRefiningStrengthFunction([[0, 1], [0, 1]])
    mesh.renew(
        random_refining_strength, [0.3, 0.4]
    )
    # f1o.evolve()
    # f1o[(0, 2)].visualize.quick()
    #
    # f1o[(0, 2)].visualize_difference_to((0, 1))

    # _ = mesh.current_representative.map
    # mesh.visualize()

    # _ = mesh.current_representative.opposite_pairs
    # f1i.cochain_switch_matrix()
    # f1o.cochain_switch_matrix()

    # f0i[(0, 1)].reduce()
    # f0o[(0, 1)].reduce()
    # f1i[(0, 1)].reduce()
    # f1o[(0, 1)].reduce()
    # f2[(0, 1)].reduce()
    # # #
    # print(f0i[(0, 1)].error())
    # print(f0o[(0, 1)].error())
    # print(f1i[(0, 1)].error())
    # print(f1o[(0, 1)].error())
    # print(f2[(0, 1)].error())

    # f0i[(0, 1)].visualize(saveto='f0i.vtk')
    # f0o[(0, 1)].visualize(saveto='f0o.vtk')
    # f1i[(0, 1)].visualize(saveto='f1i.vtk')
    # f1o[(0, 1)].visualize(saveto='f1o.vtk')
    # f2[(0, 1)].visualize(saveto='f2.vtk')
