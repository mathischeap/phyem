# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.config import _wf_term_default_simple_patterns as simple_patterns
from src.form.others import _find_form
from src.form.operators import time_derivative
from src.pde import PartialDifferentialEquations


class OrdinaryDifferentialEquationDiscretize(Frozen):
    """Ordinary Differential Equation Discretize"""

    def __init__(self, ode):
        """"""
        self._ode = ode
        self._ats = None
        self._at_instants = dict()
        self._at_intervals = dict()
        self._eq_terms = dict()
        self._freeze()

    @property
    def time_sequence(self):
        """The time sequence this discretization is working on."""
        return self._ats

    def set_time_sequence(self, ts):
        """The method of setting time sequence."""
        assert ts.__class__.__name__ == 'AbstractTimeSequence', f"I need an abstract time sequence object."
        self._ats = ts

    def define_abstract_time_instants(self, *atis):
        """For example, atis = ('k-1', 'k-0.5', 'k')."""
        assert self.time_sequence is not None, f"set the abstract time sequence first."
        for i, k in enumerate(atis):
            if k in self._at_instants:
                pass
                # warnings.warn(f"Abstract time instant {k} already exists", ExistingAbstractTimeInstantWarning)
            else:
                assert isinstance(k, str), f"{i}th abstract time instant {k} is not a string. pls use only str."
                self._at_instants[k] = self.time_sequence[k]

    def _get_abstract_time_interval(self, ks, ke):
        """make time interval [ati0, ati1]."""
        assert isinstance(ks, str) and isinstance(ke, str), f"must use string for abstract time instant."
        assert ks in self._at_instants, f"time instant {ks} is not defined"
        assert ke in self._at_instants, f"time instant {ke} is not defined"
        key = str([ks, ke])
        if key in self._at_intervals:
            return self._at_intervals[key]
        else:
            ks = self._at_instants[ks]
            ke = self._at_instants[ke]
            ati = self.time_sequence.make_time_interval(ks, ke)
            self._at_intervals[key] = ati
            return ati

    def differentiate(self, index, ks, ke, degree=1):
        """Differentiate a term at time interval [ati0, ati1] using a Gauss integrator of degree 1."""
        term = self._ode[index]
        pattern = term[2]
        ptm = term[1]
        dt = self._get_abstract_time_interval(ks, ke)
        if degree == 1:
            if pattern == simple_patterns['(pt,)']:
                bf0 = _find_form(ptm._f0._lin_repr, upon=time_derivative)
                bf0_ks = bf0 @ dt.start
                bf0_ke = bf0 @ dt.end
                bf1 = ptm._f1
                term0 = (bf0_ke - bf0_ks) / dt
                diff_term = (ptm.__class__(term0, bf1), '+')
                self._eq_terms[index] = [diff_term, ]
            else:
                raise NotImplementedError(pattern)

        else:
            raise NotImplementedError()

    def average(self, index, f, time_instants):
        """Use average at time instants `time_instants` for form `f` in term indexed `index`.
        """
        f_ = list()
        assert isinstance(time_instants, (list, tuple)), f"pls put time_instants in a list or tuple."
        for ti in time_instants:
            assert ti in self._at_instants, f"abstract time instant {ti} is not defined."
            f_.append(f @ self._at_instants[ti])

        num = len(f_)

        if num == 1:
            f_ = f_[0]
        else:
            sum_f = f_[0] + f_[1]
            if num > 2:
                for _ in f_[2:]:
                    sum_f += _
            else:
                pass
            f_ = sum_f / num

        if index not in self._eq_terms:
            term = self._ode[index][1]
            new_term, new_sign = term.replace(f, f_)
            self._eq_terms[index] = [(new_term, new_sign), ]
        else:
            term_sign = self._eq_terms[index]
            if len(term_sign) == 1:

                term, old_sign = term_sign[0]

                new_term, new_sign = term.replace(f, f_)

                if old_sign == new_sign:
                    sign = '+'
                else:
                    sign = '-'

                self._eq_terms[index] = [(new_term, sign), ]

            else:
                raise NotImplementedError()

    def __call__(self):
        """return the resulting weak formulation (of one single equation of course.)."""
        terms = ([], [])
        signs = ([], [])
        new_term_sign = self._eq_terms
        for index in self._ode:
            every_thing_about_this_term = self._ode[index]
            l_o_r = self._ode._parse_index(index)[0]
            if index in new_term_sign:
                original_sign = every_thing_about_this_term[0]
                for sign_term in new_term_sign[index]:
                    term, sign = sign_term
                    sign = self._parse_sign(original_sign, sign)
                    terms[l_o_r].append(term)
                    signs[l_o_r].append(sign)
            else:
                terms[l_o_r].append(every_thing_about_this_term[1])
                signs[l_o_r].append(every_thing_about_this_term[0])
        signs_dict = {0: signs}
        terms_dict = {0: terms}
        equation = PartialDifferentialEquations(terms_and_signs_dict=(terms_dict, signs_dict))
        return equation

    @staticmethod
    def _parse_sign(sign1, sign2):
        """-- = +, ++ = +, +- = -, -+ = - sign."""
        return '+' if sign1 == sign2 else '-'
