# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.spaces.operators import _d_to_vc, _d_ast_to_vc


class _2d_CF(Frozen):
    """"""

    def __init__(self, form, cf):
        """"""
        self._f = form
        self._check_cf(cf)
        self._freeze()

    def _check_cf(self, cf):
        """"""
        self._cf = cf

    def __call__(self, t, x, y):
        """"""
        return self._cf(t, x, y)

    @property
    def _exterior_derivative_vc_operators(self):
        """"""
        space = self._f.space.abstract

        indicator = space.indicator
        if indicator == 'Lambda':
            k = space.k
            ori = space.orientation
            return _d_to_vc(indicator, 2, 2, k, ori)
        else:
            raise NotImplementedError()

    @property
    def _codifferential_vc_operators(self):
        """"""
        space = self._f.space.abstract
        indicator = space.indicator
        if indicator == 'Lambda':
            k = space.k
            ori = space.orientation
            return _d_ast_to_vc(indicator, 2, 2, k, ori)
        else:
            raise NotImplementedError()

    def exterior_derivative(self):
        """exterior derivative."""
        vc_operator = self._exterior_derivative_vc_operators
        return getattr(self._cf, vc_operator)

    def codifferential(self):
        """codifferential"""
        sign, cd_operator = self._codifferential_vc_operators
        new_cf = getattr(self._cf, cd_operator)
        if sign == '+':
            pass
        elif sign == '-':
            new_cf = - new_cf
        else:
            raise Exception()
        return new_cf
