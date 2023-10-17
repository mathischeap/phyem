# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.config import _global_lin_repr_setting, _non_root_lin_sep
from src.form.main import Form, _global_root_forms_lin_dict
from src.config import _global_operator_lin_repr_setting
from src.config import _pde_test_form_lin_repr

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
        self._check_equations()
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

    def _check_equations(self):
        """Do a self-check after parsing terms."""
        for i in self._term_dict:
            left_terms, right_terms = self._term_dict[i]
            all_terms_of_equation_i = []
            all_terms_of_equation_i.extend(left_terms)
            all_terms_of_equation_i.extend(right_terms)

            if all([_.__class__ is Form for _ in all_terms_of_equation_i]):

                term_i0 = all_terms_of_equation_i[0]
                for term_ij in all_terms_of_equation_i[1:]:
                    assert term_i0.space == term_ij.space, \
                        f"spaces in equation #{i} do not match each other."

            elif all([
                hasattr(_, '_is_able_to_be_a_weak_term') for _ in all_terms_of_equation_i
                # has it, it must return True
            ]):
                pass
            else:
                raise Exception()

    def _pr_vc(self, figsize=(8, 6), title=None):
        """We print the pde but change all exterior derivatives to corresponding vector calculus operators."""

        from src.config import RANK, MASTER_RANK
        if RANK != MASTER_RANK:
            return
        else:
            pass

        from src.spaces.operators import _d_to_vc, _d_ast_to_vc
        from src.config import _global_operator_sym_repr_setting
        from src.config import _global_operator_lin_repr_setting
        from src.config import _non_root_lin_sep
        start, end = _non_root_lin_sep

        d_sym_repr = _global_operator_sym_repr_setting['d']
        cd_sym_repr = _global_operator_sym_repr_setting['codifferential']

        d_lin_repr = _global_operator_lin_repr_setting['d']
        cd_lin_repr = _global_operator_lin_repr_setting['codifferential']

        from src.form.others import _find_form

        number_equations = len(self._term_dict)
        symbolic = ''
        for i in self._term_dict:
            for t, forms in enumerate(self._term_dict[i]):
                if len(forms) == 0:
                    symbolic += '0'
                else:
                    for j, form in enumerate(forms):
                        sign = self._sign_dict[i][t][j]
                        form_sym_repr = form._sym_repr
                        form_lin_repr = form._lin_repr

                        if form_lin_repr.count(d_lin_repr) + form_lin_repr.count(cd_lin_repr) == 1:
                            # we only deal with the case that only one d or cd presents.
                            if d_lin_repr in form_lin_repr:
                                _ec_operator_type = 'd'
                                form_lin_repr = form_lin_repr.split(d_lin_repr)[1]
                            elif cd_lin_repr in form_lin_repr:
                                _ec_operator_type = 'cd'
                                form_lin_repr = form_lin_repr.split(cd_lin_repr)[1]
                            else:
                                raise Exception()

                            if form_lin_repr[-len(end):] == end:
                                form_lin_repr = form_lin_repr[:-len(end)]
                            else:
                                pass

                            form = _find_form(form_lin_repr)
                            space = form.space
                            space_indicator = space.indicator
                            m, n, k = space.m, space.n, space.k
                            ori = space.orientation

                            vc_operator_sym_dict = {
                                'derivative': r"\mathrm{d}",
                                'gradient': r"\nabla",
                                'curl': r"\nabla\times",
                                'rot': r"\nabla\times",
                                'divergence': r"\nabla\cdot",
                            }

                            if _ec_operator_type == 'd':
                                vc_operator = _d_to_vc(space_indicator, m, n, k, ori)
                                vc_sign = '+'
                                vc_operator = vc_operator_sym_dict[vc_operator]
                                form_sym_repr = form_sym_repr.replace(d_sym_repr, vc_operator + ' ')
                            elif _ec_operator_type == 'cd':
                                vc_sign, vc_operator = _d_ast_to_vc(space_indicator, m, n, k, ori)
                                vc_operator = vc_operator_sym_dict[vc_operator]
                                vc_operator = r"\widetilde{" + vc_operator + r"}"
                                form_sym_repr = form_sym_repr.replace(cd_sym_repr, vc_operator)
                            else:
                                raise Exception()

                            if vc_sign == '+':
                                pass
                            elif vc_sign == '-':
                                if sign == '+':
                                    sign = '-'
                                else:
                                    sign = '+'
                            else:
                                raise Exception()

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

        if len(self) > 1:
            symbolic = r"$\left\lbrace\begin{aligned}" + symbolic + r"\end{aligned}\right.$"
        else:
            symbolic = r"$\begin{aligned}" + symbolic + r"\end{aligned}$"

        if self._unknowns is None:
            ef_text = list()
            ef_text_space = list()
            for ef in self._efs:
                ef_text.append(ef._sym_repr)
                ef_text_space.append(ef.space._sym_repr)
            ef_text_space = r"$\in\left(" + r'\times '.join(ef_text_space) + r"\right)$"
            ef_text = r'for $' + r', '.join(ef_text) + r'$' + ef_text_space + ','
        else:
            ef_text_unknowns = list()
            ef_text_unknown_spaces = list()
            ef_text_others = list()
            ef_text_others_spaces = list()
            for ef in self._unknowns:
                ef_text_unknowns.append(ef._sym_repr)
                ef_text_unknown_spaces.append(ef.space._sym_repr)
            for ef in self._efs:
                if ef in self._unknowns:
                    pass
                else:
                    ef_text_others.append(ef._sym_repr)
                    ef_text_others_spaces.append(ef.space._sym_repr)
            ef_text_unknown_spaces = r"$\in\left(" + r'\times '.join(ef_text_unknown_spaces) + r"\right)$"
            ef_text_others_spaces = r"$\in\left(" + r'\times '.join(ef_text_others_spaces) + r"\right)$"
            if len(ef_text_others) == 0:

                ef_text = (r'seek unknowns: $' + r', '.join(ef_text_unknowns) +
                           r'$' + ef_text_unknown_spaces + ', such that')

            else:
                ef_text_others = r'for $' + r', '.join(ef_text_others) + r'$' + ef_text_others_spaces + ', '
                ef_text_unknowns = (r'seek $' + r', '.join(ef_text_unknowns) + r'$' +
                                    ef_text_unknown_spaces + ', such that')
                ef_text = ef_text_others + "\n" + ef_text_unknowns

        ef_text = self._mesh.manifold._manifold_text() + ef_text

        if self._bc is None or len(self._bc._valid_bcs) == 0:
            bc_text = ''
        else:
            bc_text = self.bc._bc_text()

        fig = plt.figure(figsize=figsize)
        plt.axis([0, 1, 0, 1])
        plt.axis('off')
        text = ef_text + '\n' + symbolic + bc_text
        plt.text(0.05, 0.5, text, ha='left', va='center', size=15)
        if title is None:
            pass
        else:
            plt.title(title)

        from src.config import _setting, _pr_cache
        if _setting['pr_cache']:
            _pr_cache(fig, filename='pde_vc')
        else:
            plt.tight_layout()
            plt.show(block=_setting['block'])
        return fig

    def pr(self, indexing=True, figsize=(8, 6), vc=False, title=None):
        """Print representations"""
        from src.config import RANK, MASTER_RANK
        if RANK != MASTER_RANK:
            return
        else:
            pass

        if vc is True:
            return self._pr_vc(figsize=figsize, title=title)
        else:
            pass

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
            ef_text_space = list()
            for ef in self._efs:
                ef_text.append(ef._sym_repr)
                ef_text_space.append(ef.space._sym_repr)
            ef_text_space = r"$\in\left(" + r'\times '.join(ef_text_space) + r"\right)$"
            ef_text = r'for $' + r', '.join(ef_text) + r'$' + ef_text_space + ','
        else:
            ef_text_unknowns = list()
            ef_text_unknown_spaces = list()
            ef_text_others = list()
            ef_text_others_spaces = list()
            for ef in self._unknowns:
                ef_text_unknowns.append(ef._sym_repr)
                ef_text_unknown_spaces.append(ef.space._sym_repr)
            for ef in self._efs:
                if ef in self._unknowns:
                    pass
                else:
                    ef_text_others.append(ef._sym_repr)
                    ef_text_others_spaces.append(ef.space._sym_repr)
            ef_text_unknown_spaces = r"$\in\left(" + r'\times '.join(ef_text_unknown_spaces) + r"\right)$"
            ef_text_others_spaces = r"$\in\left(" + r'\times '.join(ef_text_others_spaces) + r"\right)$"
            if len(ef_text_others) == 0:

                ef_text = (r'seek unknowns: $' + r', '.join(ef_text_unknowns) +
                           r'$' + ef_text_unknown_spaces + ', such that')

            else:
                ef_text_others = r'for $' + r', '.join(ef_text_others) + r'$' + ef_text_others_spaces + ', '
                ef_text_unknowns = (r'seek $' + r', '.join(ef_text_unknowns) + r'$' +
                                    ef_text_unknown_spaces + ', such that')
                ef_text = ef_text_others + '\n' + ef_text_unknowns

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
        if title is None:
            pass
        else:
            plt.title(title)
        from src.config import _setting, _pr_cache
        if _setting['pr_cache']:
            _pr_cache(fig, filename='pde')
        else:
            plt.tight_layout()
            plt.show(block=_setting['block'])
        return fig

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

        # -- below, we parse the test functions ----------------------------------------------
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

            j = i
            form_lin_setting = _global_lin_repr_setting['form']
            _test_lin_repr = form_lin_setting[0] + f'{j}' + _pde_test_form_lin_repr + form_lin_setting[1]
            while _test_lin_repr in _global_root_forms_lin_dict:
                j += 1
                _test_lin_repr = form_lin_setting[0] + f'{j}' + _pde_test_form_lin_repr + form_lin_setting[1]

            tf = ts.make_form(sr, f'{j}' + _pde_test_form_lin_repr)
            tfs.append(tf)

        # -------- make weak form terms ---------------------------------------------------------
        term_dict = dict()
        for i in self._term_dict:   # ith equation
            term_dict[i] = ([], [])
            for j, terms in enumerate(self._term_dict[i]):
                for k, term in enumerate(terms):

                    if term.is_root():  # we test a root-form with the test-form!

                        raw_weak_term = inner(term, tfs[i], method=test_method)

                    else:

                        multiply_lin = _global_operator_lin_repr_setting['multiply']

                        # check if we have multiplication in this pde term.
                        if multiply_lin in term._lin_repr:

                            if term._lin_repr.count(multiply_lin) == 1:

                                front_form_lin_repr, the_end_form = term._lin_repr.split(multiply_lin)

                                # below we check if we have: a scalar_parameter multiply a form
                                scalar_parameter_lin = _global_lin_repr_setting['scalar_parameter']
                                scalar_parameter_front, scalar_parameter_end = scalar_parameter_lin
                                len_scalar_parameter_front = len(scalar_parameter_front)

                                if front_form_lin_repr[:len_scalar_parameter_front] == scalar_parameter_front:

                                    sep0, sep1 = _non_root_lin_sep

                                    if the_end_form[:len(sep0)] == sep0 and the_end_form[-len(sep1):] == sep1:
                                        # - the form is not a root form
                                        root_form_lin_repr = the_end_form[len(sep0):-len(sep1)]
                                    else:
                                        root_form_lin_repr = the_end_form

                                    from src.form.others import _find_form
                                    the_form = _find_form(root_form_lin_repr)

                                    sp_lin_repr = front_form_lin_repr
                                    from src.form.parameters import _find_root_scalar_parameter

                                    root_factor = _find_root_scalar_parameter(sp_lin_repr)

                                    if root_factor is None:  # do not find a root factor.
                                        raise NotImplementedError()
                                    else:
                                        raw_weak_term = inner(the_form, tfs[i], factor=root_factor, method=test_method)

                                else:
                                    raise NotImplementedError()

                            else:
                                raise NotImplementedError()
                        else:

                            raw_weak_term = inner(term, tfs[i], method=test_method)

                    term_dict[i][j].append(raw_weak_term)

        # ------ make the weak formulation ----------------------------------------------------
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
