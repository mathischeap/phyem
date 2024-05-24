# -*- coding: utf-8 -*-
r"""
"""
from tools.functions.time_space.base import TimeSpaceFunctionBase
from functools import partial

from tools.numerical.time_space._3d.partial_derivative_as_functions import \
    NumericalPartialDerivativeTxyzFunctions


class T3dTensor(TimeSpaceFunctionBase):
    """"""

    def __init__(self, t00, t01, t02, t10, t11, t12, t20, t21, t22):
        """"""
        self._t00_ = t00
        self._t01_ = t01
        self._t02_ = t02
        self._t10_ = t10
        self._t11_ = t11
        self._t12_ = t12
        self._t20_ = t20
        self._t21_ = t21
        self._t22_ = t22
        self.__NPD00__ = None
        self.__NPD01__ = None
        self.__NPD02__ = None
        self.__NPD10__ = None
        self.__NPD11__ = None
        self.__NPD12__ = None
        self.__NPD20__ = None
        self.__NPD21__ = None
        self.__NPD22__ = None
        self._freeze()

    def __call__(self, t, x, y, z):
        return ([self._t00_(t, x, y, z), self._t01_(t, x, y, z), self._t02_(t, x, y, z)],
                [self._t10_(t, x, y, z), self._t11_(t, x, y, z), self._t12_(t, x, y, z)],
                [self._t20_(t, x, y, z), self._t21_(t, x, y, z), self._t22_(t, x, y, z)])

    def __getitem__(self, t):
        """return functions evaluated at time `t`."""
        return partial(self, t)

    def __matmul__(self, other):
        """"""
        if isinstance(other, (int, float)):
            return self[other]
        else:
            raise NotImplementedError()

    @property
    def ndim(self):
        return 3

    @property
    def shape(self):
        """a tensor"""
        return 3, 3

    @property
    def _NPD00_(self):
        if self.__NPD00__ is None:
            self.__NPD00__ = NumericalPartialDerivativeTxyzFunctions(self._t00_)
        return self.__NPD00__

    @property
    def _NPD01_(self):
        if self.__NPD01__ is None:
            self.__NPD01__ = NumericalPartialDerivativeTxyzFunctions(self._t01_)
        return self.__NPD01__

    @property
    def _NPD02_(self):
        if self.__NPD02__ is None:
            self.__NPD02__ = NumericalPartialDerivativeTxyzFunctions(self._t02_)
        return self.__NPD02__

    @property
    def _NPD10_(self):
        if self.__NPD10__ is None:
            self.__NPD10__ = NumericalPartialDerivativeTxyzFunctions(self._t10_)
        return self.__NPD10__

    @property
    def _NPD11_(self):
        if self.__NPD11__ is None:
            self.__NPD11__ = NumericalPartialDerivativeTxyzFunctions(self._t11_)
        return self.__NPD11__

    @property
    def _NPD12_(self):
        if self.__NPD12__ is None:
            self.__NPD12__ = NumericalPartialDerivativeTxyzFunctions(self._t12_)
        return self.__NPD12__

    @property
    def _NPD20_(self):
        if self.__NPD20__ is None:
            self.__NPD20__ = NumericalPartialDerivativeTxyzFunctions(self._t20_)
        return self.__NPD20__

    @property
    def _NPD21_(self):
        if self.__NPD21__ is None:
            self.__NPD21__ = NumericalPartialDerivativeTxyzFunctions(self._t21_)
        return self.__NPD21__

    @property
    def _NPD22_(self):
        if self.__NPD22__ is None:
            self.__NPD22__ = NumericalPartialDerivativeTxyzFunctions(self._t22_)
        return self.__NPD22__

    @property
    def time_derivative(self):
        pt00_pt = self._NPD00_('t')
        pt01_pt = self._NPD01_('t')
        pt02_pt = self._NPD02_('t')
        pt10_pt = self._NPD10_('t')
        pt11_pt = self._NPD11_('t')
        pt12_pt = self._NPD12_('t')
        pt20_pt = self._NPD20_('t')
        pt21_pt = self._NPD21_('t')
        pt22_pt = self._NPD22_('t')
        return self.__class__(pt00_pt, pt01_pt, pt02_pt,
                              pt10_pt, pt11_pt, pt12_pt,
                              pt20_pt, pt21_pt, pt22_pt)


if __name__ == '__main__':
    # mpiexec -n 4 python 
    pass
