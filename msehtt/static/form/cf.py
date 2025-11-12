# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.spaces.operators import _d_to_vc, _d_ast_to_vc
from tools.functions.time_space._2d.wrappers.scalar import T2dScalar
from tools.functions.time_space._2d.wrappers.vector import T2dVector

def _m2n2_0_(t, x, y):
    r""""""
    return 0 * x + 0 * y + 0 * t

class MseHttStaticFormCF(Frozen):
    """"""

    def __init__(self, f):
        """"""
        self._f = f
        self._field = None
        self._freeze()

    @property
    def field(self):
        """The continuous field."""
        return self._field

    @field.setter
    def field(self, _field):
        """"""
        if _field == 0:
            indicator, m, n = self._f.space._imn_
            if indicator == 'Lambda' and (m, n) == (2, 2):
                k = self._f.space.abstract.k
                if k == 2 or k == 0:  # this means we are receiving a zero scalar function
                    _field = T2dScalar(_m2n2_0_)
                elif k == 1:
                    _field = T2dVector(_m2n2_0_, _m2n2_0_)
                else:
                    raise Exception()
            else:
                raise NotImplementedError()
        else:
            pass
        self._field = _field

    def __getitem__(self, t):
        """"""
        return self.field[t]

    def coboundary(self):
        """alias for exterior derivative"""
        return self.exterior_derivative()

    def exterior_derivative(self):
        """Perform exterior derivative to the continuous field."""

        if self.field is None:
            raise Exception('No cf, set it first!')
        else:
            vc_operator = self._exterior_derivative_vc_operators
            d_cf = getattr(self.field, vc_operator)
            return d_cf

    def codifferential(self):
        """Perform exterior derivative to the continuous field."""

        if self.field is None:
            raise Exception('No cf, set it first!')
        else:
            sign, cd_operator = self._codifferential_vc_operators
            cd_cf = getattr(self.field, cd_operator)
            if sign == '+':
                pass
            elif sign == '-':
                cd_cf = - cd_cf
            return cd_cf

    def time_derivative(self):
        """Perform exterior derivative to the continuous field."""

        if self.field is None:
            raise Exception('No cf, set it first!')
        else:
            time_derivative = self.field.time_derivative
            return time_derivative

    @property
    def _exterior_derivative_vc_operators(self):
        """"""
        space = self._f.space.abstract
        space_indicator = space.indicator
        m, n, k = space.m, space.n, space.k
        ori = space.orientation
        return _d_to_vc(space_indicator, m, n, k, ori)

    @property
    def _codifferential_vc_operators(self):
        """"""
        space = self._f.space.abstract
        space_indicator = space.indicator
        m, n, k = space.m, space.n, space.k
        ori = space.orientation
        return _d_ast_to_vc(space_indicator, m, n, k, ori)

    def visualize(self, *args, **kwargs):
        r"""visualize this continuous form."""
        self.field.visualize(*args, **kwargs)
