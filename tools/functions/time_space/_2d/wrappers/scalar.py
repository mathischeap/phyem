# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from functools import partial
from scipy.interpolate import LinearNDInterpolator

from phyem.tools.functions.time_space.base import TimeSpaceFunctionBase

from phyem.tools.quadrature import quadrature

from phyem.tools.numerical.time_space._2d.partial_derivative_as_functions import \
    NumericalPartialDerivativeTxyFunctions, NumericalPartialDerivativeTxy

from phyem.tools.functions.time_space._2d.wrappers.helpers.scalar_add import t2d_ScalarAdd
from phyem.tools.functions.time_space._2d.wrappers.helpers.scalar_sub import t2d_ScalarSub
from phyem.tools.functions.time_space._2d.wrappers.helpers.scalar_neg import t2d_ScalarNeg
from phyem.tools.functions.time_space._2d.wrappers.helpers.scalar_mul import t2d_ScalarMultiply
from phyem.tools.functions.time_space._2d.wrappers.helpers.scalar_abs import t2d_ScalarAbs

from phyem.tools.functions.time_space._2d.wrappers.helpers.log_helper import ___LOG_HELPER___
from phyem.tools.functions.time_space._2d.wrappers.helpers.exp_helper import ___EXP_HELPER___


# noinspection PyUnusedLocal
def ___0_func2___(t, x, y):
    """"""
    return np.zeros_like(x)


class T2dScalar(TimeSpaceFunctionBase):
    """"""

    def __init__(self, s, steady=False,
                 derivative=None, second_derivative=None,
                 allowed_time_range=None, mesh=None,
                 ):
        """

        Parameters
        ----------
        s

        steady

        derivative :
            [d/dt, d/dx, d/dy]

        second_derivative :
            [dd/dt^2, dd/dx^2, dd/dxdy, dd/dydx, dd/dy^2]

        mesh :
            If it is provided, we can check and plot self using this mesh.

        """
        super().__init__(steady, allowed_time_range=allowed_time_range)
        self._s_ = s
        self.__NPD__ = None

        D = [None, None, None]  # ds_dt, ds_dx, ds_dy

        if derivative is None:
            pass
        else:
            assert isinstance(derivative, (list, tuple)) and len(derivative) == 3, \
                f"Please put df_dt, df_dx, df_dy into a list or tuple."

            for i, di in enumerate(derivative):
                if isinstance(di, (int, float)):
                    if di == 0:
                        D[i] = ___0_func2___
                    else:
                        raise NotImplementedError()
                else:
                    D[i] = di

        if self.___is_steady___:
            D[0] = ___0_func2___
        else:
            pass

        self._derivative = D
        self._dt, self._dx, self._dy = D

        second_D = [None, None, None, None, None]  # [dd/dt^2, dd/dx^2, dd/dxdy, dd/dydx, dd/dy^2]

        if second_derivative is None:
            pass
        else:
            assert isinstance(second_derivative, (list, tuple)) and len(second_derivative) == 5, \
                f"Please put 5 second derivatives: [dd/dt^2, dd/dx^2, dd/dxdy, dd/dydx, dd/dy^2] into a list or tuple."

            for i, ddi in enumerate(second_derivative):
                if isinstance(ddi, (int, float)):
                    if ddi == 0:
                        second_D[i] = ___0_func2___
                    else:
                        raise NotImplementedError()
                else:
                    second_D[i] = ddi

        self._dd_tt = second_D[0]
        self._dd_xx = second_D[1]  # dd/dx^2
        self._dd_xy = second_D[2]  # dd/dxdy
        self._dd_yx = second_D[3]  # dd/dydx
        self._dd_yy = second_D[4]  # dd/dy^2

        self._log_e_ = None
        self._exp_ = None

        if mesh is None:
            self._mesh = None
        else:
            self.mesh = mesh

        self._freeze()

    def __call__(self, t, x, y):
        return [self._s_(t, x, y), ]

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
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, _mesh):
        r""""""
        # before set the mesh, we do all checks ----------------------------------
        # 1. we first found all components to be checked -------------------------
        d_check_list = []
        dd_check_list = []
        if self._dt is not None:
            d_check_list.append('dt')
        if self._dx is not None:
            d_check_list.append('dx')
        if self._dy is not None:
            d_check_list.append('dy')
        if self._dd_tt is not None:
            dd_check_list.append('dd_tt')
        if self._dd_xx is not None:
            dd_check_list.append('dd_xx')
        if self._dd_xy is not None:
            dd_check_list.append('dd_xy')
        if self._dd_yx is not None:
            dd_check_list.append('dd_yx')
        if self._dd_yy is not None:
            dd_check_list.append('dd_yy')

        # 2. find out if we do checking ----------------------------------------------
        if len(d_check_list) != 0 or len(dd_check_list) != 0:
            do_checking = True
        else:
            do_checking = False

        # 3. prepare mesh element coo data -------------------------------------------
        X, Y = dict(), dict()
        if do_checking:
            nodes = quadrature(7, category='Gauss').quad_nodes
            xi, et = np.meshgrid(nodes, nodes, indexing='ij')
            if _mesh.__class__.__name__ == 'MseHttMeshPartial':
                ELEMENTS = _mesh.composition
            elif _mesh.__class__.__name__ == 'MseHtt_MultiGrid_MeshPartial':
                ELEMENTS = _mesh.get_level().composition
            else:
                raise NotImplementedError(_mesh.__class__.__name__)
            for i in ELEMENTS:
                element = ELEMENTS[i]
                X[i], Y[i] = element.ct.mapping(xi, et)
        else:
            pass

        # 4. do the checking ---------------------------------------------------------
        if len(d_check_list) == 0:
            pass
        else:
            # 4.1) do dt, dx, dy checking ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for i in X:
                x, y = X[i], Y[i]
                t = self._find_random_testing_time_instance_()
                npd = NumericalPartialDerivativeTxy(self._s_, t, x, y)
                if 'dt' in d_check_list:
                    assert npd.check_partial_t(self._dt)
                if 'dx' in d_check_list:
                    assert npd.check_partial_x(self._dx)
                if 'dy' in d_check_list:
                    assert npd.check_partial_y(self._dy)

        if len(dd_check_list) == 0:
            pass
        else:
            NPD_f = NumericalPartialDerivativeTxyFunctions(self._s_)
            # 4.2) do dd_tt, dd_xx, dd_xy, dd_yx, dd_yy checking ~~~~~~~~~~~~~~~~~~~~~~~~
            if 'dd_tt' in dd_check_list:
                dt_f = NPD_f('t')
                for i in X:
                    t = self._find_random_testing_time_instance_()
                    npd = NumericalPartialDerivativeTxy(dt_f, t, X[i], Y[i])
                    assert npd.check_partial_t(self._dd_tt)

            if 'dd_xx' in dd_check_list or 'dd_xy' in dd_check_list:
                dx_f = NPD_f('x')
                for i in X:
                    t = self._find_random_testing_time_instance_()
                    npd = NumericalPartialDerivativeTxy(dx_f, t, X[i], Y[i])
                    if 'dd_xx' in dd_check_list:
                        assert npd.check_partial_x(self._dd_xx)
                    if 'dd_xy' in dd_check_list:
                        assert npd.check_partial_y(self._dd_xy)

            if 'dd_yx' in dd_check_list or 'dd_yy' in dd_check_list:
                dy_f = NPD_f('y')
                for i in X:
                    t = self._find_random_testing_time_instance_()
                    npd = NumericalPartialDerivativeTxy(dy_f, t, X[i], Y[i])
                    if 'dd_yx' in dd_check_list:
                        assert npd.check_partial_x(self._dd_yx)
                    if 'dd_yy' in dd_check_list:
                        assert npd.check_partial_y(self._dd_yy)

        # =========================================================================
        self._mesh = _mesh

    def visualize(self, mesh_or_bounds=None, t=0, sampling_factor=1, **kwargs):
        """Return a visualize class for a mesh at t=`t`.

        Parameters
        ----------
        mesh_or_bounds
        t
        sampling_factor
        kwargs :
            kwargs sent to the visualizer (of dds-rws).

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
        elif mesh_or_bounds is None:
            mesh = self.mesh
        else:
            mesh = mesh_or_bounds

        assert mesh is not None, f"pls provide a mesh or set self._mesh."
        assert self.atr_check(t), f"t={t} is not allowed by the atr of this scalar."
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
        if self._dt is None:
            ps_pt = self._NPD_('t')
        else:
            ps_pt = self._dt
        return self.__class__(ps_pt, mesh=self.mesh)

    @property
    def gradient(self):
        """"""
        if self._dx is None:
            px = self._NPD_('x')
        else:
            px = self._dx

        if self._dy is None:
            py = self._NPD_('y')
        else:
            py = self._dy

        from phyem.tools.functions.time_space._2d.wrappers.vector import T2dVector

        return T2dVector(px, py, mesh=self.mesh)
    
    @property
    def curl(self):
        """R -> curl -> div -> 0"""
        if self._dx is None:
            px = self._NPD_('x')
        else:
            px = self._dx

        if self._dy is None:
            py = self._NPD_('y')
        else:
            py = self._dy

        neg_px = (- self.__class__(px))._s_

        from phyem.tools.functions.time_space._2d.wrappers.vector import T2dVector

        return T2dVector(py, neg_px, mesh=self.mesh)

    @property
    def Laplacian(self):
        r"""Return a T2dScalar instance representing laplace of self."""
        if self._dd_xx is None or self._dd_yy is None:
            u = self.gradient
            return u.divergence
        else:
            return self.__class__(t2d_ScalarAdd(self._dd_xx, self._dd_yy), mesh=self.mesh)

    def log(self, base=np.e):
        r"""return a scalar function of (t, x, y) which computes log_{base} self(t, x, t).

        Be default, we compute log_e.

        """
        if base == np.e:
            if self._log_e_ is None:
                self._log_e_ = self.__class__(___LOG_HELPER___(self._s_, base=np.e), mesh=self.mesh)
            return self._log_e_
        else:
            raise NotImplementedError()

    @property
    def exp(self):
        r"""return a scalar function of (t, x, y) which computes exp^{self(t, x, y)}."""
        if self._exp_ is None:
            self._exp_ = self.__class__(___EXP_HELPER___(self._s_), mesh=self.mesh)
        return self._exp_

    @property
    def abs(self):
        r""""""
        return self.__class__(t2d_ScalarAbs(self._s_), mesh=self.mesh)

    def convection_by(self, u):
        """We compute (u cdot nabla) of self.

        Parameters
        ----------
        u

        Returns
        -------

        """
        assert u.__class__.__name__ == "t2dVector", f"I need a t2dVector."

        if self._dx is None:
            px = self._NPD_('x')
        else:
            px = self._dx

        if self._dy is None:
            py = self._NPD_('y')
        else:
            py = self._dy

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

        elif other.__class__.__name__ == 'T2dVector':
            s0v0 = t2d_ScalarMultiply(self._s_, other._v0_)
            s0v1 = t2d_ScalarMultiply(self._s_, other._v1_)
            return other.__class__(s0v0, s0v1)

        else:
            raise NotImplementedError()

    def __rmul__(self, other):
        """other * self"""
        if isinstance(other, (int, float)):
            return self * other
        elif other.__class__.__name__ == 'T2dVector':
            return self * other
        else:
            raise NotImplementedError(other)

    def cross_product(self, other):
        """self x other"""
        from phyem.tools.functions.time_space._2d.wrappers.vector import T2dVector
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
