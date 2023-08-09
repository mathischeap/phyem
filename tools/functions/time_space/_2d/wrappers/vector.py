# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
@time: 11/3/2022 6:32 PM
"""
from tools.frozen import Frozen

from tools.functions.time_space.base import TimeSpaceFunctionBase
from functools import partial

from tools.functions.time_space._2d.wrappers.helpers.scalar_add import t2d_ScalarAdd
from tools.functions.time_space._2d.wrappers.helpers.scalar_sub import t2d_ScalarSub
from tools.functions.time_space._2d.wrappers.helpers.scalar_neg import t2d_ScalarNeg
from tools.functions.time_space._2d.wrappers.helpers.scalar_mul import t2d_ScalarMultiply

from tools.numerical.time_space._2d.partial_derivative_as_functions import \
    NumericalPartialDerivativeTxyFunctions


class T2dVector(TimeSpaceFunctionBase):
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
        self._time_derivative = None
        self._rot = None
        self._divergence = None
        self._freeze()

    def __call__(self, t, x, y):
        """Evaluate the vector at (t, x, y)"""
        return self._v0_(t, x, y), self._v1_(t, x, y)

    def __getitem__(self, t):
        """return functions evaluated at time `t`."""
        return partial(self, t)

    def visualize(self, mesh_or_bounds, t, sampling_factor=1):
        """Return a visualize class for a mesh at t=`t`.

        Parameters
        ----------
        mesh_or_bounds
        t
        sampling_factor

        Returns
        -------

        """
        if isinstance(mesh_or_bounds, (list, tuple)):  # provide a domain
            if all(isinstance(_, (int, float)) for _ in mesh_or_bounds):
                bounds = [mesh_or_bounds for _ in range(self.ndim)]
            else:
                bounds = mesh_or_bounds

            from msepy.main import _quick_mesh
            mesh = _quick_mesh(*bounds)
        else:
            mesh = mesh_or_bounds
        v = self[t]
        mesh.visualize._target(v, sampling_factor=sampling_factor)

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        """a vector"""
        return (2, )

    @property
    def _NPD0_(self):
        if self.__NPD0__ is None:
            self.__NPD0__ = NumericalPartialDerivativeTxyFunctions(self._v0_)
        return self.__NPD0__

    @property
    def _NPD1_(self):
        if self.__NPD1__ is None:
            self.__NPD1__ = NumericalPartialDerivativeTxyFunctions(self._v1_)
        return self.__NPD1__

    @property
    def time_derivative(self):
        """partial self / partial t."""
        if self._time_derivative is None:
            pv0_pt = self._NPD0_('t')
            pv1_pt = self._NPD1_('t')
            self._time_derivative = self.__class__(pv0_pt, pv1_pt)
        return self._time_derivative

    @property
    def divergence(self):
        """div(self)"""
        if self._divergence is None:
            pv0_px = self._NPD0_('x')
            pv1_py = self._NPD1_('y')
            from tools.functions.time_space._2d.wrappers.scalar import T2dScalar
            dv0 = T2dScalar(pv0_px)
            dv1 = T2dScalar(pv1_py)
            self._divergence = dv0 + dv1
        return self._divergence

    @property
    def rot(self):
        """Compute the rot of self.

        Returns
        -------

        """
        if self._rot is None:
            pv1_px = self._NPD1_('x')
            pv0_py = self._NPD0_('y')
            from tools.functions.time_space._2d.wrappers.scalar import T2dScalar
            dv0 = T2dScalar(pv1_px)
            dv1 = T2dScalar(pv0_py)
            self._rot = dv0 - dv1
        return self._rot

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

    def __mul__(self, other):
        """self * other"""
        if isinstance(other, (int, float)):

            v0_mul_number = t2d_ScalarMultiply(self._v0_, other)
            v1_mul_number = t2d_ScalarMultiply(self._v1_, other)

            return self.__class__(v0_mul_number, v1_mul_number)

        else:
            raise NotImplementedError()

    def __rmul__(self, other):
        """other * self"""
        if isinstance(other, (int, float)):
            return self * other
        else:
            raise NotImplementedError()

    def dot(self, other):
        """ self dot product with other。 So lets say self = (a, b), other = (u, v),
        self.dot(other) gives a scalar, a*u + b*v.

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

    def cross_product(self, other):
        """self x other."""
        from tools.functions.time_space._2d.wrappers.scalar import T2dScalar

        if other.__class__ is self.__class__:
            # 2-d vector x 2-d vector
            # A = [wx wy, 0]^T    B = [u v 0]^T
            # A x B = [wy*0 - 0*v,   0*u - wx*0,   wx*v - wy*u]^T = [0 0 wx*v - wy*u]^T
            wx, wy = self._v0_, self._v1_
            u, v = other._v0_, other._v1_

            V0 = t2d_ScalarMultiply(wx, v)
            V1 = t2d_ScalarMultiply(wy, u)

            V0V1 = t2d_ScalarSub(V0, V1)
            return T2dScalar(V0V1)

        elif other.__class__ is T2dScalar:
            return - other.cross_product(self)

        else:
            raise NotImplementedError()

    def outward_flux_over(self, boundary_section):
        """"""
        from msepy.mesh.boundary_section.main import MsePyBoundarySectionMesh
        if boundary_section.__class__ is MsePyBoundarySectionMesh:

            return _T2VectorFluxOverMsePyBoundarySection(self, boundary_section)

        else:
            raise NotImplementedError()


class _T2VectorFluxOverMsePyBoundarySection(Frozen):
    """It represents a scalar in 2d time-space."""

    def __init__(self, v, bs):
        """"""
        self._v = v
        self._bs = bs
        self._freeze()

    @property
    def shape(self):
        """It represents a scalar in 2d time-space."""
        return (1, )

    @property
    def ndim(self):
        """It represents a scalar in 2d time-space."""
        return 2

    def __call__(self, t, xi):
        """So we will compute the mapping of each element faces in this boundary section. This
        gives coordinates (x, y). And then we use (t, x, y) and apply it to the original
        vector, which leads to vector value (vx, vy), then the flux is  vx*nx + vy*ny
        where (nx, ny) is the outward unit norm vector of the element face.

        Parameters
        ----------
        t
        xi

        Returns
        -------

        """
        raise NotImplementedError()
