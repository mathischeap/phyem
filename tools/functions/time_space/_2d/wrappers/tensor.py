# -*- coding: utf-8 -*-
r"""
"""
from functools import partial

from phyem.tools.functions.time_space.base import TimeSpaceFunctionBase

from phyem.tools.numerical.time_space._2d.partial_derivative_as_functions import \
    NumericalPartialDerivativeTxyFunctions
from phyem.tools.functions.time_space._2d.wrappers.helpers.scalar_add import t2d_ScalarAdd
from phyem.tools.functions.time_space._2d.wrappers.helpers.scalar_sub import t2d_ScalarSub


class T2dTensor(TimeSpaceFunctionBase):
    """ Wrap four functions into a tensor class.
    """

    def __init__(
            self, t00, t01, t10, t11,
            steady=False,
            allowed_time_range=None
    ):
        """Initialize a tensor with 4 functions which take (t, x, y) as inputs.

        Parameters
        ----------
        t00
        t01
        t10
        t11
        """
        super().__init__(steady, allowed_time_range=allowed_time_range)
        self._t00_ = t00
        self._t01_ = t01
        self._t10_ = t10
        self._t11_ = t11
        self._ts_ = (
            [t00, t01],
            [t10, t11]
        )
        self.__NPD00__ = None
        self.__NPD01__ = None
        self.__NPD10__ = None
        self.__NPD11__ = None
        self._divergence = None
        self._rot = None
        self._freeze()

    def __call__(self, t, x, y):
        """Evaluate the tensor at (t, x, y)"""
        return (
            [self._t00_(t, x, y), self._t01_(t, x, y)],
            [self._t10_(t, x, y), self._t11_(t, x, y)]
        )

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
        return 2

    @property
    def shape(self):
        """a tensor"""
        return 2, 2

    @property
    def _NPD00_(self):
        if self.__NPD00__ is None:
            self.__NPD00__ = NumericalPartialDerivativeTxyFunctions(self._t00_)
        return self.__NPD00__

    @property
    def _NPD01_(self):
        if self.__NPD01__ is None:
            self.__NPD01__ = NumericalPartialDerivativeTxyFunctions(self._t01_)
        return self.__NPD01__

    @property
    def _NPD10_(self):
        if self.__NPD10__ is None:
            self.__NPD10__ = NumericalPartialDerivativeTxyFunctions(self._t10_)
        return self.__NPD10__

    @property
    def _NPD11_(self):
        if self.__NPD11__ is None:
            self.__NPD11__ = NumericalPartialDerivativeTxyFunctions(self._t11_)
        return self.__NPD11__

    @property
    def time_derivative(self):
        pt00_pt = self._NPD00_('t')
        pt01_pt = self._NPD01_('t')
        pt10_pt = self._NPD10_('t')
        pt11_pt = self._NPD11_('t')
        return self.__class__(pt00_pt, pt01_pt,
                              pt10_pt, pt11_pt)

    @property
    def divergence(self):
        if self._divergence is None:

            p00 = self._NPD00_('x')
            p01 = self._NPD01_('y')
            p10 = self._NPD10_('x')
            p11 = self._NPD11_('y')

            v0 = t2d_ScalarAdd(p00, p01)
            v1 = t2d_ScalarAdd(p10, p11)

            from phyem.tools.functions.time_space._2d.wrappers.vector import T2dVector

            self._divergence = T2dVector(v0, v1)

        return self._divergence

    @property
    def rot(self):
        if self._rot is None:

            p01_px = self._NPD01_('x')
            p00_py = self._NPD00_('y')
            p11_px = self._NPD11_('x')
            p10_py = self._NPD10_('y')

            v0 = t2d_ScalarSub(p01_px, p00_py)
            v1 = t2d_ScalarSub(p11_px, p10_py)

            from phyem.tools.functions.time_space._2d.wrappers.vector import T2dVector

            self._rot = T2dVector(v0, v1)

        return self._rot
