# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
@time: 11/3/2022 6:32 PM
"""

from tools.frozen import Frozen
from functools import partial

from tools.functions.time_space._2d.wrappers.helpers.scalar_add import t2d_ScalarAdd
from tools.functions.time_space._2d.wrappers.helpers.scalar_sub import t2d_ScalarSub
from tools.functions.time_space._2d.wrappers.helpers.scalar_neg import t2d_ScalarNeg
from tools.functions.time_space._2d.wrappers.helpers.scalar_mul import t2d_ScalarMultiply

from tools.numerical.time_space._2d.partial_derivative_as_functions import \
    NumericalPartialDerivative_txy_Functions


class T2dVector(Frozen):
    """ Wrap two functions into a vector class.
    """

    def __init__(self, v0, v1):
        """Initialize a vector with 2 functions which take (t, x, y) as inputs.

        Parameters
        ----------
        v0
        v1
        """
        self._v0_ = v0
        self._v1_ = v1
        self._vs_ = [v0, v1]
        self.__NPD0__ = None
        self.__NPD1__ = None
        self._freeze()

    def __call__(self, t, x, y):
        """Evaluate the vector at (t, x, y)"""
        return self._v0_(t, x, y), self._v1_(t, x, y)

    def __getitem__(self, t):
        """return functions evaluated at time `t`."""
        return partial(self, t)

    def visualize(self, mesh, t):
        """Return a visualize class for a mesh at t=`t`.

        Parameters
        ----------
        mesh
        t

        Returns
        -------

        """
        raise NotImplementedError()

    @property
    def ndim(self):
        return 2

    @property
    def _NPD0_(self):
        if self.__NPD0__ is None:
            self.__NPD0__ = NumericalPartialDerivative_txy_Functions(self._v0_)
        return self.__NPD0__

    @property
    def _NPD1_(self):
        if self.__NPD1__ is None:
            self.__NPD1__ = NumericalPartialDerivative_txy_Functions(self._v1_)
        return self.__NPD1__

    @property
    def time_derivative(self):
        pv0_pt = self._NPD0_('t')
        pv1_pt = self._NPD1_('t')
        return self.__class__(pv0_pt, pv1_pt)

    @property
    def divergence(self):
        pv0_px = self._NPD0_('x')
        pv1_py = self._NPD1_('y')
        from tools.functions.time_space._2d.wrappers.scalar import T2dScalar
        dv0 = T2dScalar(pv0_px)
        dv1 = T2dScalar(pv1_py)
        return dv0 + dv1

    @property
    def rot(self):
        """Compute the rot of self.

        Returns
        -------

        """
        pv1_px = self._NPD1_('x')
        pv0_py = self._NPD0_('y')
        from tools.functions.time_space._2d.wrappers.scalar import T2dScalar
        dv0 = T2dScalar(pv1_px)
        dv1 = T2dScalar(pv0_py)
        return dv0 - dv1

    def convection_by(self, u):
        """We compute (u cdot nabla) of self.

        Parameters
        ----------
        u

        Returns
        -------

        """
        assert u.__class__.__name__ == "t2dVector", f"I need a t2dVector."

        v0px = self._NPD0_('x')
        v0py = self._NPD0_('y')
        v1px = self._NPD1_('x')
        v1py = self._NPD1_('y')

        vx, vy = u._v0_, u._v1_

        v0x = t2d_ScalarMultiply(vx, v0px)
        v0y = t2d_ScalarMultiply(vy, v0py)
        v1x = t2d_ScalarMultiply(vx, v1px)
        v1y = t2d_ScalarMultiply(vy, v1py)

        return self.__class__(t2d_ScalarAdd(v0x, v0y), t2d_ScalarAdd(v1x, v1y))

    def __add__(self, other):
        """

        Parameters
        ----------
        other

        Returns
        -------

        """
        if other.__class__ is self.__class__:

            v00, v01 = self._v0_, self._v1_
            v10, v11 = other._v0_, other._v1_

            V0 = t2d_ScalarAdd(v00, v10)
            V1 = t2d_ScalarAdd(v01, v11)

            return self.__class__(V0, V1)

        else:
            raise NotImplementedError()

    def __sub__(self, other):
        """

        Parameters
        ----------
        other

        Returns
        -------

        """
        if other.__class__ is self.__class__:

            v00, v01 = self._v0_, self._v1_
            v10, v11 = other._v0_, other._v1_

            V0 = t2d_ScalarSub(v00, v10)
            V1 = t2d_ScalarSub(v01, v11)

            return self.__class__(V0, V1)

        else:
            raise NotImplementedError()

    def __neg__(self):
        v0, v1 = self._v0_, self._v1_

        neg_v0 = t2d_ScalarNeg(v0)
        neg_v1 = t2d_ScalarNeg(v1)

        return self.__class__(neg_v0, neg_v1)

    def dot(self, other):
        """ self dot product with other。 So lets say self = (a, b), other = (u, v),
        self.dot(other) gives a scalar, au + bv.

        Parameters
        ----------
        other

        Returns
        -------

        """
        if other.__class__ is self.__class__:

            v00, v01 = self._v0_, self._v1_
            v10, v11 = other._v0_, other._v1_

            V0 = t2d_ScalarMultiply(v00, v10)
            V1 = t2d_ScalarMultiply(v01, v11)

            V0V1 = t2d_ScalarAdd(V0, V1)
            from tools.functions.time_space._2d.wrappers.scalar import T2dScalar
            return T2dScalar(V0V1)

        else:
            raise NotImplementedError()
