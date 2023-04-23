# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
@time: 11/26/2022 2:56 PM
"""
import sys

if './' not in sys.path:
    sys.path.append('./')

from tools.frozen import Frozen
from src.form.main import Form

import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')

from src.wf.term.main import inner
from src.wf.main import WeakFormulation
from src.bc import BoundaryCondition


def pde(*args, **kwargs):
    """A wrapper of the PDE class."""
    return PartialDifferentialEquations(*args, **kwargs)


class PartialDifferentialEquations(Frozen):
    """partial differential equations."""

    def __init__(self, expression=None, interpreter=None, terms_and_signs_dict=None):
        if terms_and_signs_dict is None:  # provided terms and signs
            expression = self._check_expression(expression)
            interpreter = self._filter_interpreter(interpreter)
            self._parse_expression(expression, interpreter)
        else:
            assert expression is None and interpreter is None
            self._parse_terms_and_signs(terms_and_signs_dict)
        self._unknowns = None
        self._meshes, self._mesh = WeakFormulation._parse_meshes(self._term_dict)
        self._bc = None
        self._freeze()

    @staticmethod
    def _check_expression(expression):
        """"""
        if isinstance(expression, str):
            assert len(expression) > 0, "cannot be empty expression."
            expression = [expression, ]
        else:
            assert isinstance(expression, (list, tuple)), f"pls put expression in a list or tuple."
            for i, exp in enumerate(expression):
                assert isinstance(exp, str), f"expression[{i}] = {exp} is not a string."
                assert len(exp) > 0, f"expression[{i}] is empty."
        for i, equation in enumerate(expression):
            assert equation.count('=') == 1, f"expression[{i}]={equation} is wrong, can only have one '='."

        return expression

    @staticmethod
    def _filter_interpreter(interpreter):
        """"""
        new_interpreter = dict()
        for var_name in interpreter:
            if interpreter[var_name].__class__ is Form:
                new_interpreter[var_name] = interpreter[var_name]
            else:
                pass
        return new_interpreter

    def _parse_expression(self, expression, interpreter):
        """Keep upgrading this method to let it understand more equations."""
        indi_dict = dict()
        sign_dict = dict()
        term_dict = dict()
        ind_dict = dict()
        indexing = dict()
        for i, equation in enumerate(expression):

            equation = equation.replace(' ', '')  # remove all spaces
            equation = equation.replace('-', '+-')  # let all terms be connected by +

            indi_dict[i] = ([], [])  # for left terms and right terms of ith equation
            sign_dict[i] = ([], [])  # for left terms and right terms of ith equation
            term_dict[i] = ([], [])  # for left terms and right terms of ith equation
            ind_dict[i] = ([], [])  # for left terms and right terms of ith equation

            k = 0
            for j, lor in enumerate(equation.split('=')):
                local_terms = lor.split('+')

                for loc_term in local_terms:
                    if loc_term == '' or loc_term == '-':  # found empty terms, just ignore.
                        pass
                    else:
                        if loc_term == '0':
                            pass
                        else:
                            if loc_term[0] == '-':
                                assert loc_term[1:] in interpreter, f"found term {loc_term[1:]} not interpreted."
                                indi = loc_term[1:]
                                sign = '-'
                                term = interpreter[loc_term[1:]]
                            else:
                                assert loc_term in interpreter, f"found term {loc_term} not interpreted"
                                indi = loc_term
                                sign = '+'
                                term = interpreter[loc_term]

                            indi_dict[i][j].append(indi)
                            sign_dict[i][j].append(sign)
                            term_dict[i][j].append(term)
                            if j == 0:
                                index = str(i) + '-' + str(k)
                            elif j == 1:
                                index = str(i) + '-' + str(k)
                            else:
                                raise Exception()
                            k += 1
                            indexing[index] = (indi, sign, term)
                            ind_dict[i][j].append(index)

        self._indi_dict = indi_dict   # a not very import attribute. Only for print representations.
        self._sign_dict = sign_dict
        self._term_dict = term_dict   # can be form or (for example L2-inner-product- or duality-) terms
        self._ind_dict = ind_dict
        self._indexing = indexing

        efs = list()
        for i in self._term_dict:
            for terms in self._term_dict[i]:
                for term in terms:
                    if term == 0:
                        pass
                    else:
                        efs.extend(term.elementary_forms)

        self._efs = set(efs)

    def _parse_terms_and_signs(self, terms_and_signs_dict):
        """We get an equation from terms and signs."""
        self._indi_dict = None  # in this case, we will not have indi_dict; it is only used for print representations
        terms_dict, signs_dict = terms_and_signs_dict
        _ind_dict = dict()
        _indexing = dict()
        num_eq = len(terms_dict)
        for i in range(num_eq):
            assert i in terms_dict and i in signs_dict, f"numbering of equations must be 0, 1, 2, ..."
            terms = terms_dict[i]
            signs = signs_dict[i]
            _ind_dict[i] = ([], [])
            assert len(terms) == len(signs) == 2 and \
                len(terms[1]) == len(signs[1]) and len(signs[0]) == len(terms[0]), \
                f"Pls put terms and signs of {i}th equation in ([], []) and ([], []) of same length."
            k = 0
            for j, lr_terms in enumerate(terms):
                lr_signs = signs[j]
                for m, term in enumerate(lr_terms):
                    sign = lr_signs[m]
                    index = str(i) + '-' + str(k)
                    _ind_dict[i][j].append(index)
                    _indexing[index] = ('', sign, term)  # the first entry is the indicator, it is ''.
                    k += 1
        # ------ need to implement attributes below:
        self._sign_dict = signs_dict
        self._term_dict = terms_dict
        self._ind_dict = _ind_dict
        self._indexing = _indexing

        efs = list()
        for i in self._term_dict:
            for terms in self._term_dict[i]:
                for term in terms:
                    if term == 0:
                        pass
                    else:
                        efs.extend(term.elementary_forms)
        self._efs = set(efs)

    def print_representations(self, indexing=True, figsize=(8, 6)):
        """Print representations"""
        number_equations = len(self._term_dict)
        indicator = ''
        if self._indi_dict is None:
            pass
        else:
            for i in self._indi_dict:
                for t, terms in enumerate(self._indi_dict[i]):
                    if len(terms) == 0:
                        indicator += '0'
                    else:
                        for j, term in enumerate(terms):
                            term = r'\text{\texttt{' + term + '}}'
                            if indexing:
                                index = self._ind_dict[i][t][j].replace('-', r'\text{-}')
                                term = r'\underbrace{' + term + r'}_{' + \
                                       rf"{index}" + '}'
                            else:
                                pass
                            sign = self._sign_dict[i][t][j]
                            if j == 0:
                                if sign == '+':
                                    indicator += term
                                elif sign == '-':
                                    indicator += '-' + term
                                else:
                                    raise Exception()
                            else:
                                indicator += ' ' + sign + ' ' + term

                    if t == 0:
                        indicator += ' &= '
                if i < number_equations - 1:
                    indicator += r' \\ '
                else:
                    pass

        symbolic = ''
        for i in self._term_dict:
            for t, forms in enumerate(self._term_dict[i]):
                if len(forms) == 0:
                    symbolic += '0'
                else:
                    for j, form in enumerate(forms):
                        sign = self._sign_dict[i][t][j]
                        form_sym_repr = form._sym_repr
                        if indexing:
                            index = self._ind_dict[i][t][j].replace('-', r'\text{-}')
                            form_sym_repr = r'\underbrace{' + form_sym_repr + r'}_{' + \
                                rf"{index}" + '}'
                        else:
                            pass

                        if j == 0:
                            if sign == '+':
                                symbolic += form_sym_repr
                            elif sign == '-':
                                symbolic += '-' + form_sym_repr
                            else:
                                raise Exception()
                        else:
                            symbolic += ' ' + sign + ' ' + form_sym_repr

                if t == 0:
                    symbolic += ' &= '

            if i < number_equations - 1:
                symbolic += r' \\ '
            else:
                pass

        if indicator != '':
            if len(self) == 1:
                indicator = r"$\begin{aligned}" + indicator + r"\end{aligned}$"
            else:
                indicator = r"$\left\lbrace\begin{aligned}" + indicator + r"\end{aligned}\right.$"

        if len(self) > 1:
            symbolic = r"$\left\lbrace\begin{aligned}" + symbolic + r"\end{aligned}\right.$"
        else:
            symbolic = r"$\begin{aligned}" + symbolic + r"\end{aligned}$"

        if self._unknowns is None:
            ef_text = list()
            for ef in self._efs:
                ef_text.append(ef._sym_repr)
            ef_text = r'for $' + r', '.join(ef_text) + r'$,'
        else:
            ef_text_unknowns = list()
            ef_text_others = list()
            for ef in self._unknowns:
                ef_text_unknowns.append(ef._sym_repr)
            for ef in self._efs:
                if ef in self._unknowns:
                    pass
                else:
                    ef_text_others.append(ef._sym_repr)

            if len(ef_text_others) == 0:
                ef_text = r'seek unknowns: $' + r', '.join(ef_text_unknowns) + r'$, such that'
            else:
                ef_text_others = r'for $' + r', '.join(ef_text_others) + r'$, '
                ef_text_unknowns = r'seek $' + r', '.join(ef_text_unknowns) + r'$, such that'
                ef_text = ef_text_others + ef_text_unknowns

        ef_text = self._mesh.manifold._manifold_text() + ef_text

        if self._bc is None or len(self._bc._valid_bcs) == 0:
            bc_text = ''
        else:
            bc_text = self.bc._bc_text()

        fig = plt.figure(figsize=figsize)
        plt.axis([0, 1, 0, 1])
        plt.axis('off')
        if indicator == '':
            text = ef_text + '\n' + symbolic + bc_text
        else:
            text = indicator + '\n\n' + ef_text + '\n' + symbolic + bc_text

        plt.text(0.05, 0.5, text, ha='left', va='center', size=15)
        plt.tight_layout()
        plt.show()
        return fig

    def pr(self, **kwargs):
        """A wrapper of print_representations"""
        return self.print_representations(**kwargs)

    def __len__(self):
        """How many equations we have?"""
        return len(self._term_dict)

    def __getitem__(self, index):
        return self._indexing[index]

    def __iter__(self):
        """"""
        for i in self._ind_dict:
            for lri in self._ind_dict[i]:
                for index in lri:
                    yield index

    @property
    def mesh(self):
        return self._mesh

    @property
    def elementary_forms(self):
        """Return a set of root forms that this equation involves."""
        return self._efs

    @property
    def unknowns(self):
        """Unknowns"""
        return self._unknowns

    @unknowns.setter
    def unknowns(self, unknowns):
        """"""
        if self._unknowns is not None:
            f"unknowns exists; not allowed to change them."

        if len(self) == 1 and not isinstance(unknowns, (list, tuple)):
            unknowns = [unknowns, ]
        assert isinstance(unknowns, (list, tuple)), \
            f"please put unknowns in a list or tuple if there are more than 1 equations."
        assert len(unknowns) == len(self), \
            f"I have {len(self)} equations but receive {len(unknowns)} unknowns."

        for i, unknown in enumerate(unknowns):
            assert unknown.__class__ is Form and unknown.is_root(), \
                f"{i}th variable is not a root form."
            assert unknown in self._efs, f"{i}th variable = {unknown} is not an elementary form ({self._efs})."

        self._unknowns = unknowns

    def test_with(self, test_spaces, test_method='L2', sym_repr=None):
        """return a weak formulation."""
        if not isinstance(test_spaces, (list, tuple)):
            test_spaces = [test_spaces, ]
        else:
            pass

        # parse test spaces from forms if forms provided.
        _test_spaces = list()
        for i, obj in enumerate(test_spaces):
            if obj.__class__.__name__ == 'Form':
                _test_spaces.append(obj.space)
            else:
                assert obj._is_space(), f"test_spaces[{i}] is not a space."
                _test_spaces.append(obj)
        assert len(_test_spaces) == len(self), \
            f"pde has {len(self)} equations, so I need {len(self)} test spaces."

        assert self.unknowns is not None, f"Set unknowns before testing the pde."

        test_spaces = _test_spaces
        tfs = list()
        for i, ts in enumerate(test_spaces):
            unknown = None  # in case not found an unknown, will raise Error.
            for unknown in self.unknowns:
                unknown_space = unknown.space
                if ts == unknown_space:
                    break
                else:
                    unknown = None

            if sym_repr is None:
                if unknown is None:  # we do not find an unknown which is in the same space as the test form.
                    sr = r"\underline{\tau}_" + str(i)
                else:
                    assert unknown.is_root(), f"a trivial check."
                    sr = unknown._sym_repr
                    _base = sr.split('^')[0].split('_')[0]
                    sr = sr.replace(_base, r'\underline{' + _base + '}')
            else:
                assert len(sym_repr) == len(test_spaces), \
                    f"We have {len(test_spaces)} test forms, so we need {len(test_spaces)} syb_repr. " \
                    f"Now we receive {len(sym_repr)}."

                sr = sym_repr[i]

            tf = ts.make_form(sr, f'{i}th-test-form')
            tfs.append(tf)

        term_dict = dict()
        for i in self._term_dict:   # ith equation
            term_dict[i] = ([], [])
            for j, terms in enumerate(self._term_dict[i]):
                for k, term in enumerate(terms):
                    raw_weak_term = inner(term, tfs[i], method=test_method)
                    term_dict[i][j].append(raw_weak_term)

        wf = WeakFormulation(tfs, term_sign_dict=(term_dict, self._sign_dict))
        wf.unknowns = self.unknowns
        wf._bc = self._bc   # send the BC to the weak formulation.
        return wf

    @property
    def bc(self):
        """The boundary condition of pde class."""
        if self._bc is None:
            self._bc = BoundaryCondition(self._mesh)
        return self._bc


if __name__ == '__main__':
    # python src/pde.py
    import __init__ as ph
    # import phlib as ph
    ph.config.set_embedding_space_dim(3)
    manifold = ph.manifold(3)
    mesh = ph.mesh(manifold)

    ph.space.set_mesh(mesh)
    O0 = ph.space.new('Lambda', 0)
    O1 = ph.space.new('Lambda', 1)
    O2 = ph.space.new('Lambda', 2)
    O3 = ph.space.new('Lambda', 3)

    w = O1.make_form(r'\omega^1', "vorticity1")
    u = O2.make_form(r'u^2', r"velocity2")
    f = O2.make_form(r'f^2', r"body-force")
    P = O3.make_form(r'P^3', r"total-pressure3")

    # ph.list_spaces()
    # ph.list_forms(globals())

    wXu = w.wedge(ph.Hodge(u))

    dsP = ph.codifferential(P)
    dsu = ph.codifferential(u)
    du = ph.d(u)

    du_dt = ph.time_derivative(u)

    ph.list_forms(globals())
    # du_dt.print_representations()

    exp = [
        'du_dt + wXu - dsP = f',
        'w = dsu',
        'du = 0',
    ]

    pde = ph.pde(exp, locals())
    pde.unknowns = [u, w, P]
    pde.pr()
