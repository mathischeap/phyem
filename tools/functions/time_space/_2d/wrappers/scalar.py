# -*- coding: utf-8 -*-
r"""
"""
import sys

import numpy as np

if './' not in sys.path:
    sys.path.append('./')

from tools.functions.time_space.base import TimeSpaceFunctionBase

from tools.numerical.time_space._2d.partial_derivative_as_functions import \
    NumericalPartialDerivativeTxyFunctions

from tools.functions.time_space._2d.wrappers.helpers.scalar_add import t2d_ScalarAdd
from tools.functions.time_space._2d.wrappers.helpers.scalar_sub import t2d_ScalarSub
from tools.functions.time_space._2d.wrappers.helpers.scalar_neg import t2d_ScalarNeg
from tools.functions.time_space._2d.wrappers.helpers.scalar_mul import t2d_ScalarMultiply

from functools import partial
from scipy.interpolate import LinearNDInterpolator


class T2dScalar(TimeSpaceFunctionBase):
    """"""

    def __init__(self, s):
        """"""
        self._s_ = s
        self.__NPD__ = None
        self._freeze()

    def __call__(self, t, x, y):
        return [self._s_(t, x, y), ]

    def __getitem__(self, t):
        """return functions evaluated at time `t`."""
        return partial(self, t)

    def visualize(self, mesh_or_bounds, t, sampling_factor=1, **kwargs):
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
        mesh.visualize._target(v, sampling_factor=sampling_factor, **kwargs)

    @property
    def ndim(self):
        """"""
        return 2

    @property
    def shape(self):
        """a scalar"""
        return (1, )

    @property
    def _NPD_(self):
        """"""
        if self.__NPD__ is None:
            self.__NPD__ = NumericalPartialDerivativeTxyFunctions(self._s_)
        return self.__NPD__

    @property
    def time_derivative(self):
        """"""
        ps_pt = self._NPD_('t')
        return self.__class__(ps_pt)

    @property
    def gradient(self):
        """"""

        px = self._NPD_('x')
        py = self._NPD_('y')

        from tools.functions.time_space._2d.wrappers.vector import T2dVector

        return T2dVector(px, py)
    
    @property
    def curl(self):
        """R -> curl -> div -> 0"""
        px = self._NPD_('x')
        py = self._NPD_('y')

        neg_px = (- self.__class__(px))._s_

        from tools.functions.time_space._2d.wrappers.vector import T2dVector

        return T2dVector(py, neg_px)

    def convection_by(self, u):
        """We compute (u cdot nabla) of self.

        Parameters
        ----------
        u

        Returns
        -------

        """
        assert u.__class__.__name__ == "t2dVector", f"I need a t2dVector."

        px = self._NPD_('x')
        py = self._NPD_('y')

        vx, vy = u._v0_, u._v1_

        sx = t2d_ScalarMultiply(vx, px)
        sy = t2d_ScalarMultiply(vy, py)

        return self.__class__(t2d_ScalarAdd(sx, sy))

    def __add__(self, other):
        """"""
        if other.__class__ is self.__class__:

            s0_add_s1 = t2d_ScalarAdd(self._s_, other._s_)

            return self.__class__(s0_add_s1)

        else:
            raise NotImplementedError()

    def __sub__(self, other):
        """"""
        if other.__class__ is self.__class__:

            s0_sub_s1 = t2d_ScalarSub(self._s_, other._s_)

            return self.__class__(s0_sub_s1)

        else:
            raise NotImplementedError()

    def __neg__(self):
        """"""

        neg = t2d_ScalarNeg(self._s_)

        return self.__class__(neg)

    def __mul__(self, other):
        """self * other"""
        if other.__class__ is self.__class__:
            s0_mul_s1 = t2d_ScalarMultiply(self._s_, other._s_)
            return self.__class__(s0_mul_s1)

        elif isinstance(other, (int, float)):
            s0_mul_s1 = t2d_ScalarMultiply(self._s_, other)
            return self.__class__(s0_mul_s1)

        else:
            raise NotImplementedError()

    def __rmul__(self, other):
        """other * self"""
        if isinstance(other, (int, float)):
            return self * other
        else:
            raise NotImplementedError()

    def cross_product(self, other):
        """self x other"""
        from tools.functions.time_space._2d.wrappers.vector import T2dVector
        if other.__class__ is T2dVector:
            # scalar x vector
            # A is self (scalar), B is vector (other)
            # so, A = [0 0 w]^T, B = [u, v, 0]^T
            # cp_term = A X B = [-wv wu 0]^T
            w = self._s_
            u, v = other._v0_, other._v1_
            V0 = t2d_ScalarMultiply(w, v)
            V1 = t2d_ScalarMultiply(w, u)
            V0 = t2d_ScalarNeg(V0)
            return T2dVector(V0, V1)
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

        caller = _MergeCaller1(
            self, msepy_form,
            [N_RM, S_RM, W_RM, E_RM],
            ffs, tfs,
            reconstruction_nodes
        )

        return self.__class__(caller)


class _MergeCaller1(object):
    """"""
    def __init__(self, scalar, form, rms, form_faces, total_faces, nodes):
        """"""
        self._scalar = scalar
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

    def __call__(self, t, x, y):
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
                value = self._scalar(t, self._x_list[i], self._y_list[i])[0]

            v.append(value)

        v = np.concatenate(v)
        interpolate = LinearNDInterpolator(self._points, v)
        v = interpolate(x, y)
        return v


if __name__ == '__main__':
    # mpiexec -n 4 python tools/functions/time_space/_2d/wrappers/scalar.py
    def f0(t, x, y):
        return x + y + t

    def f1(t, x, y):
        return x * y + t

    s0 = T2dScalar(f0)
    s1 = T2dScalar(f1)
