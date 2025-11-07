# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from tools.functions.time_space.base import TimeSpaceFunctionBase

from tools.numerical.time_space._3d.partial_derivative_as_functions import \
    NumericalPartialDerivativeTxyzFunctions

from functools import partial

from tools.functions.time_space._3d.wrappers.helpers.scalar_mul import t3d_ScalarMultiply
from tools.functions.time_space._3d.wrappers.helpers._3scalars_add import t3d_3ScalarAdd
from tools.functions.time_space._3d.wrappers.helpers.scalar_add import t3d_ScalarAdd
from tools.functions.time_space._3d.wrappers.helpers.scalar_sub import t3d_ScalarSub
from tools.functions.time_space._3d.wrappers.helpers.scalar_neg import t3d_ScalarNeg


# noinspection PyUnusedLocal
def ___0_func___(t, x, y, z):
    """"""
    return np.zeros_like(x)


class T3dScalar(TimeSpaceFunctionBase):
    """"""

    def __init__(self, s, steady=False, derivative=None):
        """

        Parameters
        ----------
        s
        steady :
            This scalar is independent of time. So df/dt = 0.
        derivative
        """
        super().__init__(steady)
        if isinstance(s, (int, float)) and s == 0:
            self.___is_zero___ = True
            s = ___0_func___
        else:
            self.___is_zero___ = False
            pass

        self._s_ = s
        self.__NPD__ = None

        D = [None, None, None, None]  # ds_dt, ds_dx, ds_dy, ds_dz
        if derivative is None:
            pass
        else:
            assert isinstance(derivative, (list, tuple)) and len(derivative) == 4, \
                f"Please put df_dt, df_dx, df_dy, df_dz into a list or tuple."

            for i, di in enumerate(derivative):
                if isinstance(di, (int, float)):
                    if di == 0:
                        D[i] = ___0_func___
                    else:
                        raise NotImplementedError()
                else:
                    D[i] = di

        if self.___is_steady___:
            D[0] = ___0_func___
        else:
            pass

        self._derivative = D
        self._dt, self._dx, self._dy, self._dz = D

        self._freeze()

    def __call__(self, t, x, y, z):
        return [self._s_(t, x, y, z), ]

    def __getitem__(self, t):
        """return functions evaluated at time `t`."""
        return partial(self, t)

    def __matmul__(self, other):
        """"""
        if isinstance(other, (int, float)):
            return self[other]
        else:
            raise NotImplementedError()

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
        return 3

    @property
    def shape(self):
        return (1, )

    @property
    def _NPD_(self):
        if self.__NPD__ is None:
            self.__NPD__ = NumericalPartialDerivativeTxyzFunctions(self._s_)
        return self.__NPD__

    @property
    def time_derivative(self):
        if self._dt is None:
            ps_pt = self._NPD_('t')
        else:
            ps_pt = self._dt
        return self.__class__(ps_pt)

    @property
    def gradient(self):
        """"""
        from tools.functions.time_space._3d.wrappers.vector import T3dVector
        if self.___is_zero___:
            return T3dVector(0, 0, 0)
        else:
            if self._dx is None:
                px = self._NPD_('x')
            else:
                px = self._dx

            if self._dy is None:
                py = self._NPD_('y')
            else:
                py = self._dy

            if self._dz is None:
                pz = self._NPD_('z')
            else:
                pz = self._dz

            return T3dVector(px, py, pz)

    def convection_by(self, u):
        """We compute (u cdot nabla) of self.

        Parameters
        ----------
        u

        Returns
        -------

        """
        assert u.__class__.__name__ == "t3dVector", f"I need a t3dVector."

        if self._dx is None:
            px = self._NPD_('x')
        else:
            px = self._dx

        if self._dy is None:
            py = self._NPD_('y')
        else:
            py = self._dy

        if self._dz is None:
            pz = self._NPD_('z')
        else:
            pz = self._dz

        vx, vy, vz = u._v0_, u._v1_, u._v2_

        sx = t3d_ScalarMultiply(vx, px)
        sy = t3d_ScalarMultiply(vy, py)
        sz = t3d_ScalarMultiply(vz, pz)

        return self.__class__(t3d_3ScalarAdd(sx, sy, sz))

    def __add__(self, other):
        """"""
        if other.__class__ is self.__class__:

            s0_add_s1 = t3d_ScalarAdd(self._s_, other._s_)

            return self.__class__(s0_add_s1)

        else:
            raise NotImplementedError()

    def __sub__(self, other):
        """"""
        if other.__class__ is self.__class__:

            s0_sub_s1 = t3d_ScalarSub(self._s_, other._s_)

            return self.__class__(s0_sub_s1)

        else:
            raise NotImplementedError()

    def __neg__(self):
        """"""
        if self.___is_zero___:
            return self
        else:
            neg = t3d_ScalarNeg(self._s_)

            return self.__class__(neg)

    def __mul__(self, other):
        """"""
        if other.__class__ is self.__class__:
            s0_mul_s1 = t3d_ScalarMultiply(self._s_, other._s_)
            return self.__class__(s0_mul_s1)

        else:
            raise NotImplementedError()
