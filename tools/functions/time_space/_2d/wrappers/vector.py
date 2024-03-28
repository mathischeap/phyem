# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen

from tools.functions.time_space.base import TimeSpaceFunctionBase
from functools import partial

from tools.functions.time_space._2d.wrappers.helpers.scalar_add import t2d_ScalarAdd
from tools.functions.time_space._2d.wrappers.helpers.scalar_sub import t2d_ScalarSub
from tools.functions.time_space._2d.wrappers.helpers.scalar_neg import t2d_ScalarNeg
from tools.functions.time_space._2d.wrappers.helpers.scalar_mul import t2d_ScalarMultiply
from tools.functions.time_space._2d.wrappers.helpers.norm_helper import NormHelper2DVector

from tools.numerical.time_space._2d.partial_derivative_as_functions import \
    NumericalPartialDerivativeTxyFunctions


from scipy.interpolate import LinearNDInterpolator


# noinspection PyUnusedLocal
def _0_function(t, x, y):
    return np.zeros_like(x)


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
        if v0 == 0:
            v0 = _0_function

        if v1 == 0:
            v1 = _0_function

        self._v0_ = v0
        self._v1_ = v1
        self._vs_ = [v0, v1]
        self.__NPD0__ = None
        self.__NPD1__ = None
        self._time_derivative = None
        self._rot = None
        self._divergence = None
        self._gradient = None
        self._curl = None
        self._norm = None
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
    def gradient(self):
        """Gives a 2 by 2 tensor."""
        if self._gradient is None:
            p0_px = self._NPD0_('x')
            p0_py = self._NPD0_('y')
            p1_px = self._NPD1_('x')
            p1_py = self._NPD1_('y')

            from tools.functions.time_space._2d.wrappers.tensor import T2dTensor

            self._gradient = T2dTensor(p0_px, p0_py, p1_px, p1_py)

        return self._gradient
    
    @property
    def curl(self):
        if self._curl is None:
            p0_px = self._NPD0_('x')
            p0_py = self._NPD0_('y')
            p1_px = self._NPD1_('x')
            p1_py = self._NPD1_('y')

            neg_p0_px = t2d_ScalarNeg(p0_px)
            neg_p1_px = t2d_ScalarNeg(p1_px)

            from tools.functions.time_space._2d.wrappers.tensor import T2dTensor

            self._curl = T2dTensor(p0_py, neg_p0_px, p1_py, neg_p1_px)

        return self._curl
    
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

    @property
    def norm(self):
        """

        Returns
        -------

        """
        if self._norm is None:
            norm = NormHelper2DVector(self._v0_, self._v1_)
            from tools.functions.time_space._2d.wrappers.scalar import T2dScalar
            self._norm = T2dScalar(norm)
        return self._norm

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

    def x(self, other):
        """shortcut of `cross_product`."""
        return self.cross_product(other)

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

    def tensor_product(self, other):
        """self otimes other"""

        if other.__class__ is self.__class__:
            from tools.functions.time_space._2d.wrappers.tensor import T2dTensor

            # self = (u, v)
            # other = (a ,b)
            # self otimes other = ([ua, ub], [va, vb])

            u, v = self._v0_, self._v1_
            a, b = other._v0_, other._v1_

            t00 = t2d_ScalarMultiply(u, a)
            t01 = t2d_ScalarMultiply(u, b)
            t10 = t2d_ScalarMultiply(v, a)
            t11 = t2d_ScalarMultiply(v, b)

            return T2dTensor(t00, t01, t10, t11)

        else:
            raise NotImplementedError()

    def _merge_msepy_form_over_boundary(self, msepy_form, boundary_msepy_manifold, sampling_factor=1):
        """For the boundary of `msepy_form.mesh`, on the `boundary_msepy_manifold` part, we return
        the reconstruction of `msepy_form`, on else part, we return the evaluation of self!

        Parameters
        ----------
        msepy_form
        boundary_msepy_manifold

        Returns
        -------

        """
        from msepy.main import base
        from msepy.manifold.main import MsePyManifold
        from msepy.mesh.main import MsePyMesh
        mesh = msepy_form.mesh
        assert mesh.__class__ is MsePyMesh, f"mesh must be a MsePyMesh!"
        assert boundary_msepy_manifold.__class__ is MsePyManifold, f"boundary_msepy_manifold must be a msepy manifold!"
        assert mesh.n == boundary_msepy_manifold.n + 1 == 2, f"boundary_msepy_manifold must be of dimension n-1=1."
        meshes = base['meshes']
        boundary_manifold = mesh.abstract.manifold.boundary()
        form_boundary_mesh = None
        total_boundary_mesh = None
        for sym_repr in meshes:
            msepy_mesh = meshes[sym_repr]
            if msepy_mesh.manifold is boundary_msepy_manifold:
                form_boundary_mesh = msepy_mesh
            else:
                pass
            if msepy_mesh.manifold.abstract is boundary_manifold:
                total_boundary_mesh = msepy_mesh
            else:
                pass

        assert form_boundary_mesh is not None, f"we must have found a msepy boundary mesh!"
        assert form_boundary_mesh.base is mesh, f"boundary_msepy_manifold is not a boundary of the mesh."
        assert total_boundary_mesh is not None, f"We must have found the total boundary mesh."

        tfs = total_boundary_mesh.faces
        ffs = form_boundary_mesh.faces

        samples = 350 * sampling_factor
        num_total_faces = len(tfs)
        num_nodes = int(samples / num_total_faces)
        if num_nodes < 5:
            num_nodes = 5
        else:
            pass

        reconstruction_nodes = np.linspace(-1, 1, num_nodes)
        ones = np.array([1])

        N_nodes = [-ones, reconstruction_nodes]
        S_nodes = [ones, reconstruction_nodes]
        W_nodes = [reconstruction_nodes, -ones]
        E_nodes = [reconstruction_nodes, ones]

        N_elements, S_elements, W_elements, E_elements = list(), list(), list(), list()

        for element, m, n in zip(*ffs._elements_m_n):
            if m == 0:
                if n == 0:
                    N_elements.append(element)
                elif n == 1:
                    S_elements.append(element)
                else:
                    raise Exception()
            elif m == 1:
                if n == 0:
                    W_elements.append(element)
                elif n == 1:
                    E_elements.append(element)
                else:
                    raise Exception()
            else:
                raise Exception()

        N_RM = msepy_form.reconstruction_matrix(*N_nodes, element_range=N_elements)
        S_RM = msepy_form.reconstruction_matrix(*S_nodes, element_range=S_elements)
        W_RM = msepy_form.reconstruction_matrix(*W_nodes, element_range=W_elements)
        E_RM = msepy_form.reconstruction_matrix(*E_nodes, element_range=E_elements)

        caller = _MergeCaller2dVector(
            self, msepy_form,
            [N_RM, S_RM, W_RM, E_RM],
            ffs, tfs,
            reconstruction_nodes
        )

        return self.__class__(caller._vx, caller._vy)


class _MergeCaller2dVector(Frozen):
    """"""
    def __init__(self, vector, form, rms, form_faces, total_faces, nodes):
        """"""
        self._vector = vector
        self._form = form
        self._rms = rms
        self._ffs = form_faces
        self._tfs = total_faces
        self._nodes = nodes
        x_list = list()
        y_list = list()
        for i in self._tfs:
            face = self._tfs[i]
            x, y = face.ct.mapping(self._nodes)
            x_list.append(x)
            y_list.append(y)
        self._x_list = x_list
        self._y_list = y_list

        x_interpolate = np.concatenate(self._x_list)
        y_interpolate = np.concatenate(self._y_list)
        points = np.array([x_interpolate, y_interpolate]).T
        self._points = points
        self._freeze()

    def _vx(self, t, x, y):
        """"""
        form_cochain = self._form.cochain[t]

        v = list()
        for i in self._tfs:
            face = self._tfs[i]
            if face in self._ffs:
                element = face._element
                m, n = face._m, face._n
                if m == 0:
                    if n == 0:
                        rm = self._rms[0][element]   # N
                    elif n == 1:
                        rm = self._rms[1][element]   # S
                    else:
                        raise Exception()
                elif m == 1:
                    if n == 0:
                        rm = self._rms[2][element]   # W
                    elif n == 1:
                        rm = self._rms[3][element]   # E
                    else:
                        raise Exception()
                else:
                    raise Exception()
                value = rm[0] @ form_cochain.local[element]
            else:
                value = self._vector._v0_(t, self._x_list[i], self._y_list[i])

            v.append(value)

        v = np.concatenate(v)
        interpolate = LinearNDInterpolator(self._points, v)
        vx = interpolate(x, y)
        return vx

    def _vy(self, t, x, y):
        """"""
        form_cochain = self._form.cochain[t]

        v = list()
        for i in self._tfs:
            face = self._tfs[i]
            if face in self._ffs:
                element = face._element
                m, n = face._m, face._n
                if m == 0:
                    if n == 0:
                        rm = self._rms[0][element]   # N
                    elif n == 1:
                        rm = self._rms[1][element]   # S
                    else:
                        raise Exception()
                elif m == 1:
                    if n == 0:
                        rm = self._rms[2][element]   # W
                    elif n == 1:
                        rm = self._rms[3][element]   # E
                    else:
                        raise Exception()
                else:
                    raise Exception()
                value = rm[1] @ form_cochain.local[element]
            else:
                value = self._vector._v1_(t, self._x_list[i], self._y_list[i])

            v.append(value)

        v = np.concatenate(v)
        interpolate = LinearNDInterpolator(self._points, v)
        return interpolate(x, y)
