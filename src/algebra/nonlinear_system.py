# -*- coding: utf-8 -*-
r"""
"""

from tools.frozen import Frozen
from src.algebra.linear_system import LinearSystem
from src.wf.term.ap import TermNonLinearMDAAlgebraicProxy


class NonLinearSystem(Frozen):
    """"""

    def __init__(self, ls, left_nonlinear_terms):
        """A nonlinear system is composite of a linear system and some nonlinear terms. The nonlinear terms
        are all on the left hand side of the equations.

        Parameters
        ----------
        ls
        left_nonlinear_terms
        """
        assert ls.__class__ is LinearSystem, f"nonlinear system is based on a LinearSystem."
        self._ls = ls
        self._n_terms, self._n_signs = self._check_nonlinear_terms(left_nonlinear_terms)
        self._freeze()

    def _check_nonlinear_terms(self, nonlinear_terms):
        """Nonlinear terms should be put in a dict, for example:

            {
                0: a
                1: [b, ]
                2: [c, d, e, ...]
            }
        where a, b, c, d, e, ... are the nonlinear terms (for example, instances of TermNonLinearAlgebraicProxy).

        This dict means in the first equation, we have nonlinear term `a` in the second one, we have `b`; in
        the third one, we have ``c``, ``d``, ``e``, ...

        Parameters
        ----------
        nonlinear_terms

        Returns
        -------

        """
        num_nonlinear_terms = 0
        all_terms, all_signs = nonlinear_terms
        assert len(all_terms) == self._ls.b_shape[0], f"nonlinear terms shape wrong!"

        for i, terms in enumerate(all_terms):
            for j, term in enumerate(terms):
                num_nonlinear_terms += 1
                if term.__class__ is TermNonLinearMDAAlgebraicProxy:
                    pass
                else:
                    raise NotImplementedError()
                assert all_signs[i][j] in ('+', '-'), f"sing={all_signs[i][j]} is wring"

        self._num_nonlinear_terms = num_nonlinear_terms
        return nonlinear_terms

    @property
    def num_nonlinear_terms(self):
        """I have how many nonlinear terms?"""
        return self._num_nonlinear_terms
