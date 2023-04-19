# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
@time: 2/18/2023 10:18 PM
"""
import sys

if './' not in sys.path:
    sys.path.append('./')
from src.tools.frozen import Frozen
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')
from src.wf.td import TemporalDiscretization
from src.bc import BoundaryCondition
from src.wf.derive import _Derive
from src.wf.ap.main import AlgebraicProxy
from src.wf.mp.main import MatrixProxy


class WeakFormulation(Frozen):
    """Weak Formulation."""

    def __init__(self, test_forms, term_sign_dict=None, expression=None, merge=None):
        """

        Parameters
        ----------
        term_sign_dict
        test_forms
        expression
        """
        if term_sign_dict is not None:
            assert expression is None
            assert merge is None
            self._parse_term_sign_dict(term_sign_dict, test_forms)

        elif expression is not None:
            assert term_sign_dict is None
            assert merge is None
            self._parse_expression(expression, test_forms)

        elif merge is not None:  # merge multiple weak formulations
            assert term_sign_dict is None
            assert expression is None
            self._initialize_through_merging(merge, test_forms)

        else:
            raise Exception()

        self._meshes, self._mesh = self._parse_meshes(self._term_dict)
        self._consistence_checker()
        self._unknowns = None
        self._derive = None
        self._td = None
        self._bc = None
        self._freeze()

    def _parse_term_sign_dict(self, term_sign_dict, test_forms):
        """"""
        term_dict, sign_dict = term_sign_dict
        ind_dict = dict()
        indexing = dict()
        num_eq = len(term_dict)
        for i in range(num_eq):   # ith equation
            assert i in term_dict and i in sign_dict, f"numbering of equations must be 0, 1, 2, ..."
            ind_dict[i] = ([], [])
            k = 0
            for j, terms in enumerate(term_dict[i]):
                for m in range(len(terms)):
                    index = str(i) + '-' + str(k)
                    k += 1
                    indexing[index] = (sign_dict[i][j][m], term_dict[i][j][m])
                    ind_dict[i][j].append(index)

        self._test_forms = test_forms
        self._term_dict = term_dict
        self._sign_dict = sign_dict
        self._ind_dict = ind_dict
        self._indexing = indexing

    def _parse_expression(self, expression, test_forms):
        """"""
        raise NotImplementedError()

    def _initialize_through_merging(self, merge, test_forms):
        """"""
        term_dict = dict()
        sign_dict = dict()
        no = 0
        for i in merge:
            eqi = merge[i]
            if eqi.__class__.__name__ == 'PartialDifferentialEquations':
                tdi, sdi = eqi._term_dict, eqi._sign_dict
                for j in tdi:
                    term_dict[no] = tdi[j]
                    sign_dict[no] = sdi[j]
                    no += 1

            elif isinstance(eqi, dict):
                tdi, sdi = eqi['_term_dict'], eqi['_sign_dict']
                term_dict[no] = tdi
                sign_dict[no] = sdi
                no += 1

            else:
                raise NotImplementedError()

        for i in term_dict:
            for j, terms in enumerate(term_dict[i]):
                for k, term in enumerate(terms):
                    assert term._is_able_to_be_a_weak_term, f"term[{i}][{j}][{k}] = {term} " \
                                                            f"is not suitable for a weak formulation."
                    sign = sign_dict[i][j][k]
                    assert sign in ('+', '-'), f"sign[{i}][{j}][{k}] = {sign} illegal."
        self._parse_term_sign_dict([term_dict, sign_dict], test_forms)

    @classmethod
    def _parse_meshes(cls, term_dict):
        """"""
        meshes = list()
        for i in term_dict:
            for terms in term_dict[i]:
                for t in terms:
                    mesh = t.mesh
                    if mesh not in meshes:
                        meshes.append(mesh)
                    else:
                        pass

        num_meshes = len(meshes)
        assert num_meshes > 0, f"we need at least one mesh."

        if num_meshes == 1:
            mesh = meshes[0]  # config #1: one mesh found, then we are probably dealing with periodic manifold.
            return meshes, mesh  # RETURN config #1
        else:
            mesh_dim_dict = dict()
            for m in meshes:
                ndim = m.ndim
                if ndim not in mesh_dim_dict:
                    mesh_dim_dict[ndim] = list()
                else:
                    pass
                mesh_dim_dict[ndim].append(m)

            # below we analyze more than one mesh ---------------
            if len(mesh_dim_dict) == 2:
                larger_ndim = max(mesh_dim_dict.keys())
                lower_ndim = min(mesh_dim_dict.keys())
                if lower_ndim == larger_ndim - 1 and \
                        len(mesh_dim_dict[larger_ndim]) == 1 and \
                        len(mesh_dim_dict[lower_ndim]) == 1:
                    mesh = mesh_dim_dict[larger_ndim][0]   # config #2: found one mesh, and its boundary mesh.
                    # This is the most common case.
                    boundary_mesh = mesh_dim_dict[lower_ndim][0]
                    assert mesh.manifold.boundary() == boundary_mesh.manifold, f"must be the case. Safety check."
                    return meshes, mesh  # RETURN config #2
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

    def _consistence_checker(self):
        """We do consistence check here and parse properties like mesh and so on."""
        ts = list()
        for tf in self._test_forms:
            ts.append(tf.space)
        self._test_spaces = ts

        efs = set()
        for i in self._term_dict:   # ith equation
            for terms in self._term_dict[i]:
                for term in terms:
                    efs.update(term.elementary_forms)

        self._efs = efs

    def __repr__(self):
        """Customize the __repr__."""
        super_repr = super().__repr__().split('object')[1]
        if self.unknowns is None:
            unknown_sym_repr = '[...]'
        else:
            unknown_sym_repr = [f._sym_repr for f in self.unknowns]
        return f"<WeakFormulation of {unknown_sym_repr}" + super_repr

    def __getitem__(self, item):
        """"""
        assert item in self._indexing, \
            f"index: '{item}' is illegal, do `print_representations(indexing=True)` " \
            f"to check indices of all terms."
        return self._indexing[item]

    def __iter__(self):
        """"""
        for i in self._ind_dict:
            for lri in self._ind_dict[i]:
                for index in lri:
                    yield index

    def __len__(self):
        """"""
        return len(self._term_dict)

    def __contains__(self, item):
        return item in self._indexing

    def _parse_index(self, index):
        """"""
        ith_equation, k = index.split('-')
        ith_equation = int(ith_equation)
        k = int(k)
        left_terms = self._term_dict[ith_equation][0]
        number_left_terms = len(left_terms)
        if k < number_left_terms:
            l_o_r = 0  # left
            ith_term = k
        else:
            l_o_r = 1
            ith_term = k - number_left_terms
        return ith_equation, l_o_r, ith_term

    @property
    def elementary_forms(self):
        """Return a set of root forms that this equation involves."""
        return self._efs

    @property
    def unknowns(self):
        """The unknowns of this weak formulation"""
        return self._unknowns

    @unknowns.setter
    def unknowns(self, unknowns):
        """The unknowns of this weak formulation"""
        if self._unknowns is not None:
            f"unknowns exists; not allowed to change them."

        if len(self) == 1 and not isinstance(unknowns, (list, tuple)):
            unknowns = [unknowns, ]
        assert isinstance(unknowns, (list, tuple)), \
            f"please put unknowns in a list or tuple if there are more than 1 equations."
        assert len(unknowns) == len(self), \
            f"I have {len(self)} equations but receive {len(unknowns)} unknowns."

        for i, unknown in enumerate(unknowns):
            assert unknown.__class__.__name__ == 'Form' and unknown.is_root(), \
                f"{i}th variable is not a root form."
            assert unknown in self._efs, f"{i}th variable is not an elementary form."

        self._unknowns = unknowns

    @property
    def test_forms(self):
        return self._test_forms

    def pr(self, indexing=True):
        """Print the representations"""
        seek_text = self._mesh.manifold._manifold_text()
        if self.unknowns is None:
            seek_text += r'for $\left('
            form_sr_list = list()
            space_sr_list = list()
            for ef in self._efs:
                if ef not in self._test_forms:
                    form_sr_list.append(rf' {ef._sym_repr}')
                    space_sr_list.append(rf"{ef.space._sym_repr}")
                else:
                    pass
            seek_text += ','.join(form_sr_list)
            seek_text += r'\right) \in '
            seek_text += r'\times '.join(space_sr_list)
            seek_text += '$, \n'
        else:
            given_text = r'for'
            for ef in self._efs:
                if ef not in self.unknowns and ef not in self._test_forms:
                    given_text += rf' ${ef._sym_repr} \in {ef.space._sym_repr}$, '
            if given_text == r'for':
                seek_text += r'seek $\left('
            else:
                seek_text += given_text + '\n'
                seek_text += r'seek $\left('
            form_sr_list = list()
            space_sr_list = list()
            for un in self.unknowns:
                form_sr_list.append(rf' {un._sym_repr}')
                space_sr_list.append(rf"{un.space._sym_repr}")
            seek_text += ','.join(form_sr_list)
            seek_text += r'\right) \in '
            seek_text += r'\times '.join(space_sr_list)
            seek_text += '$, such that\n'
        symbolic = ''
        number_equations = len(self._term_dict)
        for i in self._term_dict:
            for t, terms in enumerate(self._term_dict[i]):
                if len(terms) == 0:
                    symbolic += '0'
                else:

                    for j, term in enumerate(terms):
                        sign = self._sign_dict[i][t][j]
                        term = self._term_dict[i][t][j]

                        term_sym_repr = term._sym_repr

                        if indexing:
                            index = self._ind_dict[i][t][j].replace('-', r'\text{-}')
                            term_sym_repr = r'\underbrace{' + term_sym_repr + r'}_{' + \
                                rf"{index}" + '}'
                        else:
                            pass

                        if j == 0:
                            if sign == '+':
                                symbolic += term_sym_repr
                            elif sign == '-':
                                symbolic += '-' + term_sym_repr
                            else:
                                raise Exception()
                        else:
                            symbolic += ' ' + sign + ' ' + term_sym_repr

                if t == 0:
                    symbolic += ' &= '

            symbolic += r'\quad &&\forall ' + self._test_forms[i]._sym_repr + r'\in ' + \
                self._test_spaces[i]._sym_repr

            if i < number_equations - 1:
                symbolic += r',\\'
            else:
                symbolic += '.'

        symbolic = r"$\left\lbrace\begin{aligned}" + symbolic + r"\end{aligned}\right.$"
        if self._bc is None or len(self.bc) == 0:
            bc_text = ''
        else:
            bc_text = self.bc._bc_text()

        if indexing:
            figsize = (12, 3 * len(self._term_dict))
        else:
            figsize = (12, 3 * len(self._term_dict))

        plt.figure(figsize=figsize)
        plt.axis([0, 1, 0, 1])
        plt.axis('off')
        plt.text(0.05, 0.5, seek_text + symbolic + bc_text, ha='left', va='center', size=15)
        plt.tight_layout()
        plt.show()

    @property
    def derive(self):
        """The derivations that to be applied to this current weak formulation."""
        if self._derive is None:
            self._derive = _Derive(self)
        return self._derive

    @property
    def bc(self):
        """The boundary condition of pde class."""
        if self._bc is None:
            self._bc = BoundaryCondition(self._mesh)
        return self._bc

    @property
    def td(self):
        """temporal discretization."""
        if self._td is None:
            self._td = TemporalDiscretization(self)
        return self._td

    def ap(self):
        """Do not cache it. Make it in real time"""
        return AlgebraicProxy(self)

    def mp(self):
        """Do not cache it. Make it in real time"""
        return MatrixProxy(self)


if __name__ == '__main__':
    # python src/wf/main.py

    import __init__ as ph
    # import phlib as ph
    # ph.config.set_embedding_space_dim(3)
    manifold = ph.manifold(3)
    mesh = ph.mesh(manifold)

    ph.space.set_mesh(mesh)
    O0 = ph.space.new('Lambda', 0)
    O1 = ph.space.new('Lambda', 1)
    O2 = ph.space.new('Lambda', 2)
    O3 = ph.space.new('Lambda', 3)

    # ph.list_spaces()
    # ph.list_meshes()

    w = O1.make_form(r'\omega^1', "vorticity1")
    u = O2.make_form(r'u^2', "velocity2")
    f = O2.make_form(r'f^2', "body-force")
    P = O3.make_form(r'P^3', "total-pressure3")

    wXu = w.wedge(ph.Hodge(u))
    dsP = ph.codifferential(P)
    dsu = ph.codifferential(u)
    du = ph.d(u)
    du_dt = ph.time_derivative(u)

    # ph.list_forms(globals())

    exp = [
        'du_dt + wXu - dsP = f',
        'w = dsu',
        'du = 0',
    ]
    pde = ph.pde(exp, globals())
    pde.unknowns = [u, w, P]
    # pde.pr(indexing=True)

    wf = pde.test_with([O2, O1, O3], sym_repr=[r'v^2', r'w^1', r'q^3'])

    wf = wf.derive.integration_by_parts('0-2')
    wf = wf.derive.integration_by_parts('1-1')

    wf = wf.derive.rearrange(
        {
            0: '0, 1, 2 = 4, 3',
            1: '1, 0 = 2',
            2: ' = 0',
        }
    )

    wf.pr()

    # i = 0
    # terms = wf._term_dict[i]
    # signs = wf._sign_dict[i]
    #
    # ode_i = ph.ode(terms_and_signs=[terms, signs])
    # ode_i.print_representations()
    # ph.list_forms()

    # td = wf.td
    # ap = wf.ap
