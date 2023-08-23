# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.form.others import _find_form
from src.config import _wf_term_default_simple_patterns as _simple_patterns
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')
from src.ode.discretize import OrdinaryDifferentialEquationDiscretize


def ode(*args, **kwargs):
    """A wrapper of the ODE class."""
    return OrdinaryDifferentialEquation(*args, **kwargs)


class OrdinaryDifferentialEquation(Frozen):
    """ODE about t only. Only deal with real number valued terms."""

    def __init__(self, expression=None, terms_and_signs=None):
        """"""
        if expression is None:
            assert terms_and_signs is not None, f"must provide one of `expression` or `term_sign_dict`."
            self._parse_terms_and_signs(terms_and_signs)
        elif terms_and_signs is None:
            self._parse_expression(expression)
        else:
            raise Exception(f'Pls provide only one of `expression` or `term_sign_dict`.')
        self._analyze_terms()
        self._discretize = None
        self._freeze()

    def _parse_terms_and_signs(self, terms_and_signs):
        """"""
        terms, signs = terms_and_signs
        assert len(terms) == len(signs) == 2, f"terms, signs length wrong, must be 2 " \
                                              f"(representing left and right terms)."

        left_terms, right_terms = terms
        left_signs, right_signs = signs

        if not isinstance(left_terms, (list, tuple)):
            left_terms = [left_terms, ]
        if not isinstance(right_terms, (list, tuple)):
            right_terms = [right_terms, ]
        if not isinstance(left_signs, (list, tuple)):
            left_signs = [left_signs, ]
        if not isinstance(right_signs, (list, tuple)):
            right_signs = [right_signs, ]

        assert len(left_terms) == len(left_signs), f"left terms length is not equal to left signs length."
        assert len(right_terms) == len(right_signs), f"right terms length is not equal to right signs length."

        partial_t_orders = ([], [])
        partial_t_terms = ([], [])
        other_terms = ([], [])
        pattern = ([], [])

        efs = set()
        number_valid_terms = 0
        for i, terms in enumerate([left_terms, right_terms]):
            for j, term in enumerate(terms):
                sign = signs[i][j]
                assert sign in ('+', '-'), rf"sign[{i}][{j}] = {sign} is wrong."
                assert term._is_real_number_valued(), f"{j}th term in left terms is not real number valued."
                efs.update(term.elementary_forms)
                valid_pattern, order = None, None
                sp = term._simple_pattern
                if sp in self._recognized_pattern():
                    assert valid_pattern is None, f"find more than 1 valid pattern for term {term}."
                    valid_pattern = sp
                    order = self._recognized_pattern()[sp]
                if valid_pattern is None:  # this is not a partial t term.
                    partial_t_orders[i].append(None)
                    partial_t_terms[i].append(None)
                    other_terms[i].append(term)
                    pattern[i].append(None)
                else:
                    number_valid_terms += 1
                    partial_t_orders[i].append(order)
                    partial_t_terms[i].append(term)
                    other_terms[i].append(None)
                    pattern[i].append(valid_pattern)

        self._num_pt_terms = number_valid_terms  # can be 0
        self._signs = signs
        self._order = partial_t_orders
        self._pterm = partial_t_terms
        self._other = other_terms
        self._pattern = pattern   # each time can only have one pattern.
        self._efs = efs

    def _parse_expression(self, expression):
        """"""
        raise NotImplementedError(f"must set `sign`, `_order`, `pterm`, `_other`, "
                                  f"`_pattern` `_efs` properties.")

    @classmethod
    def _recognized_pattern(cls):
        """"""
        return {
            # pattern indicator      : order
            _simple_patterns['(pt,)']: 1,  # inner product of time-derivative of a root-s-form and another s-form.
        }

    def _analyze_terms(self):
        """analyze terms"""
        _about = list()
        overall_order = list()
        k = 0
        indexing = dict()
        ind = ([], [])
        for i, terms in enumerate(self._pterm):
            for j, term in enumerate(terms):
                if term is not None:
                    order = self._order[i][j]
                    overall_order.append(order)
                    pattern = self._pattern[i][j]
                    assert pattern is not None, f"trivial check."
                    if pattern == '(partial_t root-f, f)':
                        f0 = term._f0
                        f0_lin_repr = f0._lin_repr
                        rf_lin_repr = r'\textsf{' + f0_lin_repr.split(r'\textsf{')[1]
                        rf = _find_form(rf_lin_repr)
                        _about.append(rf)
                    else:
                        raise NotImplementedError()

                else:
                    order = 0
                    term = self._other[i][j]
                    pattern = None

                index = str(k)
                indexing[index] = (self._signs[i][j], term, pattern, order)
                ind[i].append(index)
                k += 1

        _about = list(set(_about))
        assert len(_about) <= 1, f"this ode should have no or only about a single root-form."
        if len(_about) == 1:
            self._about = _about[0]
            self._overall_order = max(overall_order)
        else:
            self._about = None
            self._overall_order = None
        self._ind = ind
        self._indexing = indexing

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        ele_form_repr = set([form._sym_repr for form in self._efs])
        return f"<ODE of {self._about._sym_repr} for {ele_form_repr}" + super_repr

    def __contains__(self, index):
        """contains"""
        return index in self._indexing

    def __getitem__(self, index):
        """getitem"""
        if isinstance(index, int):
            index = str(index)
        else:
            pass
        assert index in self._indexing, f"Cannot find term indexed {index} for {self}."
        return self._indexing[index]

    def __iter__(self):
        """Iter"""
        for index in self._indexing:
            yield index

    def _parse_index(self, index):
        """parse index"""
        k = int(index)
        left_terms = self._signs[0]
        number_left_terms = len(left_terms)
        if k < number_left_terms:
            l_o_r = 0  # left
            ith_term = k
        else:
            l_o_r = 1
            ith_term = k - number_left_terms
        return l_o_r, ith_term

    @property
    def elementary_forms(self):
        """The elementary forms."""
        return self._efs

    def pr(self, indexing=True, figsize=(12, 4)):
        """print representations"""
        from src.config import RANK, MASTER_RANK
        if RANK != MASTER_RANK:
            return
        else:
            pass

        sym = r'\noindent Time derivative of: '
        sym += rf'${self._about._sym_repr}$, '
        if len(self.elementary_forms) > 1:  # as it must contain the `about` form.
            sym += 'elementary forms: '
            for ef in self.elementary_forms:
                if ef is self._about:
                    pass
                else:
                    sym += rf'${ef._sym_repr}$, '

        sym += r'\\\\$'
        for i, terms in enumerate(self._pterm):
            if len(terms) == 0:
                sym += '0'
            else:
                for j, term in enumerate(terms):
                    ptm = term
                    otm = self._other[i][j]
                    sign = self._signs[i][j]
                    if j == 0 and sign == '+':
                        sign = ''

                    if ptm is None:
                        sym_term = otm._sym_repr
                    else:
                        sym_term = ptm._sym_repr

                    if indexing:
                        index = self._ind[i][j]
                        sym_term = r"\underbrace{" + sym_term + '}_{' + index + '}'
                    else:
                        pass

                    sym += sign + sym_term

            if i == 0:
                sym += '='

        sym += '$'
        fig = plt.figure(figsize=figsize)
        plt.axis([0, 1, 0, 1])
        plt.axis('off')
        plt.text(0.05, 0.5, sym, ha='left', va='center', size=15)
        plt.tight_layout()
        from src.config import _setting
        plt.show(block=_setting['block'])
        return fig

    @property
    def discretize(self):
        """To discretize this ode."""
        if self._discretize is None:
            self._discretize = OrdinaryDifferentialEquationDiscretize(self)
        return self._discretize
