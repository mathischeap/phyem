# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from tools.functions.time_space.base import TimeSpaceFunctionBase
from functools import partial

from tools.numerical.time_space._3d.partial_derivative_as_functions import \
    NumericalPartialDerivativeTxyzFunctions

from tools.functions.time_space._3d.wrappers.helpers._3scalars_add import t3d_3ScalarAdd
from tools.functions.time_space._3d.wrappers.helpers.scalar_sub import t3d_ScalarSub
from tools.functions.time_space._3d.wrappers.helpers.scalar_mul import t3d_ScalarMultiply
from tools.functions.time_space._3d.wrappers.helpers.scalar_add import t3d_ScalarAdd
from tools.functions.time_space._3d.wrappers.helpers.scalar_neg import t3d_ScalarNeg


# noinspection PyUnusedLocal
def ___0_func___(t, x, y, z):
    return np.zeros_like(x)


class T3dVector(TimeSpaceFunctionBase):
    """ Wrap three functions into a vector class.
    """

    def __init__(self, v0, v1, v2, steady=False, Jacobian_matrix=None, time_derivative=None):
        """Initialize a vector with 3 functions which take (t, x, y, z) as inputs.

        Parameters
        ----------
        v0
        v1
        v2
        steady :
            If it is steady, then this vector is independent of t.
        Jacobian_matrix :
            We can provide Jacobian matrix (3 by 3). We can give some components of it.
            For the missing ones, just give None. For example,

            Jacobian_matrix = (
                [None, None, None],
                [dv_dx, None, None],
                [dw_dx, dw_dy, None]
            )

            We receive dv_dx, dw_dx, and dw_dy. For other components, we will use numerical
            approach to compute them.

            dw_dx, dw_dy, and dv_dx are all functions taking (t, x, y, z) as inputs.

        time_derivative :
            Like Jacobian_matrix, we can provide the analytical expressions of the
            time derivative.

            For example,

            time_derivative = (None, dv_dt, dw_dt)

            we provide dv_dt and dw_dt and leave du_dx for numerical approach.

            dv_dt and dw_dt are both functions taking (t, x, y, z) as inputs.

        """
        super().__init__(steady)   # if it is steady, it is independent of t!
        if isinstance(v0, (int, float)) and v0 == 0:
            v0 = ___0_func___
        else:
            pass
        if isinstance(v1, (int, float)) and v1 == 0:
            v1 = ___0_func___
        else:
            pass
        if isinstance(v2, (int, float)) and v2 == 0:
            v2 = ___0_func___
        else:
            pass

        self._v0_ = v0
        self._v1_ = v1
        self._v2_ = v2
        self._vs_ = [v0, v1, v2]
        self.__NPD0__ = None
        self.__NPD1__ = None
        self.__NPD2__ = None

        JM = [
            [None, None, None],  # du_dx, du_dy, du_dz
            [None, None, None],  # dv_dx, dv_dy, dv_dz
            [None, None, None],  # dw_dx, dw_dy, dw_dz
        ]
        if Jacobian_matrix is None:
            pass
        else:
            assert isinstance(Jacobian_matrix, (list, tuple)) and len(Jacobian_matrix) == 3, \
                f"Jacobian_matrix must be a 3 by 3 object (like a tuple or list)"
            for i, J_ in enumerate(Jacobian_matrix):
                assert len(J_) == 3, (f"Jacobian_matrix must be a 3 by 3 object (like a tuple or list), "
                                      f"len(J[{i},:])={len(J_)}, is not 3.")
                for j, Jij in enumerate(J_):
                    if isinstance(Jij, (int, float)):
                        if Jij == 0:
                            JM[i][j] = ___0_func___
                        else:
                            raise NotImplementedError()
                    else:
                        JM[i][j] = Jacobian_matrix[i][j]

        self._JM = JM
        self._du_dx = JM[0][0]
        self._du_dy = JM[0][1]
        self._du_dz = JM[0][2]
        self._dv_dx = JM[1][0]
        self._dv_dy = JM[1][1]
        self._dv_dz = JM[1][2]
        self._dw_dx = JM[2][0]
        self._dw_dy = JM[2][1]
        self._dw_dz = JM[2][2]

        if self.___is_steady___:

            self._td = (___0_func___, ___0_func___, ___0_func___)

        else:
            if time_derivative is None:
                time_derivative = (None, None, None)
            else:
                assert len(time_derivative) == 3, \
                    f"must provide three components of time_derivative representing (du_dt, dv_dt, dw_dt)."

            self._td = time_derivative

        self._du_dt, self._dv_dt, self._dw_dt = self._td

        self._freeze()

    def __call__(self, t, x, y, z):
        """Evaluate the vector at (t, x, y, z)"""
        return self._v0_(t, x, y, z), self._v1_(t, x, y, z), self._v2_(t, x, y, z)

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
        return (3, )

    @property
    def _NPD0_(self):
        if self.__NPD0__ is None:
            self.__NPD0__ = NumericalPartialDerivativeTxyzFunctions(self._v0_)
        return self.__NPD0__

    @property
    def _NPD1_(self):
        if self.__NPD1__ is None:
            self.__NPD1__ = NumericalPartialDerivativeTxyzFunctions(self._v1_)
        return self.__NPD1__

    @property
    def _NPD2_(self):
        if self.__NPD2__ is None:
            self.__NPD2__ = NumericalPartialDerivativeTxyzFunctions(self._v2_)
        return self.__NPD2__

    @property
    def time_derivative(self):
        if self._du_dt is None:
            pv0_pt = self._NPD0_('t')
        else:
            pv0_pt = self._du_dt

        if self._dv_dt is None:
            pv1_pt = self._NPD1_('t')
        else:
            pv1_pt = self._dv_dt

        if self._dw_dt is None:
            pv2_pt = self._NPD2_('t')
        else:
            pv2_pt = self._dw_dt

        return self.__class__(pv0_pt, pv1_pt, pv2_pt)

    @property
    def divergence(self):
        if self._du_dx is None:
            pv0_px = self._NPD0_('x')
        else:
            pv0_px = self._du_dx

        if self._dv_dy is None:
            pv1_py = self._NPD1_('y')
        else:
            pv1_py = self._dv_dy

        if self._dw_dz is None:
            pv2_pz = self._NPD2_('z')
        else:
            pv2_pz = self._dw_dz

        scalar_function = t3d_3ScalarAdd(pv0_px, pv1_py, pv2_pz)

        from tools.functions.time_space._3d.wrappers.scalar import T3dScalar

        return T3dScalar(scalar_function)

    @property
    def curl(self):
        """The curl of a 3d vector. Let's say self is (u, v, w):

        (pw/py - pv/pz, pu/pz - pw/px, pv/px - pu/py)

        Returns
        -------

        """
        if self._dw_dy is None:
            pw_py = self._NPD2_('y')
        else:
            pw_py = self._dw_dy

        if self._dw_dx is None:
            pw_px = self._NPD2_('x')
        else:
            pw_px = self._dw_dx

        if self._du_dz is None:
            pu_pz = self._NPD0_('z')
        else:
            pu_pz = self._du_dz

        if self._du_dy is None:
            pu_py = self._NPD0_('y')
        else:
            pu_py = self._du_dy

        if self._dv_dx is None:
            pv_px = self._NPD1_('x')
        else:
            pv_px = self._dv_dx

        if self._dv_dz is None:
            pv_pz = self._NPD1_('z')
        else:
            pv_pz = self._dv_dz

        v0 = t3d_ScalarSub(pw_py, pv_pz)
        v1 = t3d_ScalarSub(pu_pz, pw_px)
        v2 = t3d_ScalarSub(pv_px, pu_py)

        return self.__class__(v0, v1, v2)

    def convection_by(self, u):
        """We compute (u cdot nabla) of self where u is another t3d vector.

        Parameters
        ----------
        u

        Returns
        -------

        """
        assert u.__class__.__name__ == "t3dVector", f"I need a t3dVector."

        if self._du_dx is None:
            v0px = self._NPD0_('x')
        else:
            v0px = self._du_dx

        if self._du_dy is None:
            v0py = self._NPD0_('y')
        else:
            v0py = self._du_dy

        if self._du_dz is None:
            v0pz = self._NPD0_('z')
        else:
            v0pz = self._du_dz

        if self._dv_dx is None:
            v1px = self._NPD1_('x')
        else:
            v1px = self._dv_dx

        if self._dv_dy is None:
            v1py = self._NPD1_('y')
        else:
            v1py = self._dv_dy

        if self._dv_dz:
            v1pz = self._NPD1_('z')
        else:
            v1pz = self._dv_dz

        if self._dw_dx is None:
            v2px = self._NPD2_('x')
        else:
            v2px = self._dw_dx

        if self._dw_dy is None:
            v2py = self._NPD2_('y')
        else:
            v2py = self._dw_dy

        if self._dw_dz is None:
            v2pz = self._NPD2_('z')
        else:
            v2pz = self._dw_dz

        vx, vy, vz = u._v0_, u._v1_, u._v2_

        v0x = t3d_ScalarMultiply(vx, v0px)
        v0y = t3d_ScalarMultiply(vy, v0py)
        v0z = t3d_ScalarMultiply(vz, v0pz)
        v1x = t3d_ScalarMultiply(vx, v1px)
        v1y = t3d_ScalarMultiply(vy, v1py)
        v1z = t3d_ScalarMultiply(vz, v1pz)
        v2x = t3d_ScalarMultiply(vx, v2px)
        v2y = t3d_ScalarMultiply(vy, v2py)
        v2z = t3d_ScalarMultiply(vz, v2pz)

        return self.__class__(t3d_3ScalarAdd(v0x, v0y, v0z),
                              t3d_3ScalarAdd(v1x, v1y, v1z),
                              t3d_3ScalarAdd(v2x, v2y, v2z))

    def __add__(self, other):
        """

        Parameters
        ----------
        other

        Returns
        -------

        """
        if other.__class__ is self.__class__:

            v00, v01, v02 = self._v0_, self._v1_, self._v2_
            v10, v11, v12 = other._v0_, other._v1_, other._v2_

            V0 = t3d_ScalarAdd(v00, v10)
            V1 = t3d_ScalarAdd(v01, v11)
            V2 = t3d_ScalarAdd(v02, v12)

            return self.__class__(V0, V1, V2)

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

            v00, v01, v02 = self._v0_, self._v1_, self._v2_
            v10, v11, v12 = other._v0_, other._v1_, other._v2_

            V0 = t3d_ScalarSub(v00, v10)
            V1 = t3d_ScalarSub(v01, v11)
            V2 = t3d_ScalarSub(v02, v12)

            return self.__class__(V0, V1, V2)

        else:
            raise NotImplementedError()

    def __neg__(self):
        v0, v1, v2 = self._v0_, self._v1_, self._v2_

        neg_v0 = t3d_ScalarNeg(v0)
        neg_v1 = t3d_ScalarNeg(v1)
        neg_v2 = t3d_ScalarNeg(v2)

        return self.__class__(neg_v0, neg_v1, neg_v2)

    def __mul__(self, other):
        """self * other"""
        if isinstance(other, (int, float)):

            v0_mul_number = t3d_ScalarMultiply(self._v0_, other)
            v1_mul_number = t3d_ScalarMultiply(self._v1_, other)
            v2_mul_number = t3d_ScalarMultiply(self._v2_, other)

            return self.__class__(v0_mul_number, v1_mul_number, v2_mul_number)

        else:
            raise NotImplementedError()

    def __rmul__(self, other):
        """other * self"""
        if isinstance(other, (int, float)):
            return self * other
        else:
            raise NotImplementedError()

    def dot(self, other):
        """`self` dot product with `other`. So lets say self = (a, b, c), other = (u, v, w),
        self.dot(other) gives a scalar, a*u + b*v + c*w.

        Parameters
        ----------
        other

        Returns
        -------

        """
        if other.__class__ is self.__class__:

            v00, v01, v02 = self._v0_, self._v1_, self._v2_
            v10, v11, v12 = other._v0_, other._v1_, other._v2_

            V0 = t3d_ScalarMultiply(v00, v10)
            V1 = t3d_ScalarMultiply(v01, v11)
            V2 = t3d_ScalarMultiply(v02, v12)

            V0V1V2 = t3d_3ScalarAdd(V0, V1, V2)
            from tools.functions.time_space._3d.wrappers.scalar import T3dScalar
            return T3dScalar(V0V1V2)

        else:
            raise NotImplementedError()

    def cross_product(self, other):
        """"""
        if other.__class__ is self.__class__:

            a, b, c = self._v0_, self._v1_, self._v2_
            u, v, w = other._v0_, other._v1_, other._v2_

            bw = t3d_ScalarMultiply(b, w)
            cu = t3d_ScalarMultiply(c, u)
            av = t3d_ScalarMultiply(a, v)
            cv = t3d_ScalarMultiply(c, v)
            aw = t3d_ScalarMultiply(a, w)
            bu = t3d_ScalarMultiply(b, u)

            V0 = self.__class__(bw, cu, av)
            V1 = self.__class__(cv, aw, bu)

            return V0 - V1

        else:
            raise NotImplementedError()
