# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from functools import partial

from phyem.tools.functions.time_space.base import TimeSpaceFunctionBase

from phyem.tools.numerical.time_space._3d.partial_derivative_as_functions import \
    NumericalPartialDerivativeTxyzFunctions, NumericalPartialDerivativeTxyz

from phyem.tools.quadrature import quadrature

from phyem.tools.functions.time_space._3d.wrappers.helpers.scalar_mul import t3d_ScalarMultiply
from phyem.tools.functions.time_space._3d.wrappers.helpers._3scalars_add import t3d_3ScalarAdd
from phyem.tools.functions.time_space._3d.wrappers.helpers.scalar_add import t3d_ScalarAdd
from phyem.tools.functions.time_space._3d.wrappers.helpers.scalar_add3 import t3d_ScalarAdd3
from phyem.tools.functions.time_space._3d.wrappers.helpers.scalar_sub import t3d_ScalarSub
from phyem.tools.functions.time_space._3d.wrappers.helpers.scalar_neg import t3d_ScalarNeg
from phyem.tools.functions.time_space._3d.wrappers.helpers.scalar_abs3 import t3d_ScalarAbs

from phyem.tools.functions.time_space._3d.wrappers.helpers.log_helper3 import ___LOG_HELPER_3___
from phyem.tools.functions.time_space._3d.wrappers.helpers.exp_helper3 import ___EXP_HELPER_3___


# noinspection PyUnusedLocal
def ___0_func___(t, x, y, z):
    """"""
    return np.zeros_like(x)


class T3dScalar(TimeSpaceFunctionBase):
    """"""

    def __init__(
            self, s, steady=False,
            derivative=None,
            second_derivative=None,
            allowed_time_range=None,
            mesh=None,
    ):
        """

        Parameters
        ----------
        s
        steady :
            This scalar is independent of time. So df/dt = 0.
        derivative
        second_derivative :
            [
                dtdt,
                dxdx, dxdy, dxdz,
                dydx, dydy, dydz,
                dzdx, dzdy, dzdz,
            ]

        mesh :
            If it is provided, we can check and visualize self using this mesh.

        """
        super().__init__(steady, allowed_time_range=allowed_time_range)
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

        second_D = [None, None, None, None, None, None, None, None, None, None]

        if second_derivative is None:
            pass
        else:
            assert isinstance(second_derivative, (list, tuple)) and len(second_derivative) == 10, \
                (f"Please put 10 second derivatives: [dtdt, dxdx, dxdy, dxdz, dydx, dydy, dydz, dzdx, dzdy, dzdz,] "
                 f"into a list or tuple.")

            for i, ddi in enumerate(second_derivative):
                if isinstance(ddi, (int, float)):
                    if ddi == 0:
                        second_D[i] = ___0_func___
                    else:
                        raise NotImplementedError()
                else:
                    second_D[i] = ddi

        self._dd_tt = second_D[0]
        self._dd_xx = second_D[1]  # dxdx
        self._dd_xy = second_D[2]  # dxdy
        self._dd_xz = second_D[3]  # dxdz
        self._dd_yx = second_D[4]  # dydx
        self._dd_yy = second_D[5]  # dydy
        self._dd_yz = second_D[6]  # dydz
        self._dd_zx = second_D[7]  # dzdx
        self._dd_zy = second_D[8]  # dzdy
        self._dd_zz = second_D[9]  # dzdz

        if self.___is_steady___:
            D[0] = ___0_func___
        else:
            pass

        self._derivative = D
        self._dt, self._dx, self._dy, self._dz = D

        if mesh is None:
            self._mesh = None
        else:
            self.mesh = mesh

        self._log_e_ = None
        self._exp_ = None

        self._freeze()

    def __call__(self, t, x, y, z):
        return [self._s_(t, x, y, z), ]

    def __getitem__(self, t):
        """return functions evaluated at time `t`."""
        return partial(self, t)

    def __matmul__(self, other):
        """self @ other"""
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
        if self._dz is not None:
            d_check_list.append('dz')

        if self._dd_tt is not None:
            dd_check_list.append('dd_tt')

        if self._dd_xx is not None:
            dd_check_list.append('dd_xx')
        if self._dd_xy is not None:
            dd_check_list.append('dd_xy')
        if self._dd_xz is not None:
            dd_check_list.append('dd_xz')

        if self._dd_yx is not None:
            dd_check_list.append('dd_yx')
        if self._dd_yy is not None:
            dd_check_list.append('dd_yy')
        if self._dd_yz is not None:
            dd_check_list.append('dd_yz')

        if self._dd_zx is not None:
            dd_check_list.append('dd_zx')
        if self._dd_zy is not None:
            dd_check_list.append('dd_zy')
        if self._dd_zz is not None:
            dd_check_list.append('dd_zz')

        # 2. find out if we do checking ----------------------------------------------
        if len(d_check_list) != 0:
            do_checking = True
        else:
            do_checking = False

        # 3. prepare mesh element coo data -------------------------------------------
        X, Y, Z = dict(), dict(), dict()
        if do_checking:
            nodes = quadrature(5, category='Gauss').quad_nodes
            xi, et, sg = np.meshgrid(nodes, nodes, nodes, indexing='ij')
            if _mesh.__class__.__name__ == 'MseHttMeshPartial':
                ELEMENTS = _mesh.composition
            else:
                raise NotImplementedError()
            for i in ELEMENTS:
                element = ELEMENTS[i]
                X[i], Y[i], Z[i] = element.ct.mapping(xi, et, sg)
        else:
            pass

        # 4. do the checking ---------------------------------------------------------
        if len(d_check_list) == 0:
            pass
        else:
            # 4.1) do dt, dx, dy checking ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for i in X:
                x, y, z = X[i], Y[i], Z[i]
                t = self._find_random_testing_time_instance_()
                npd = NumericalPartialDerivativeTxyz(self._s_, t, x, y, z)
                if 'dt' in d_check_list:
                    assert npd.check_partial_t(self._dt)
                if 'dx' in d_check_list:
                    assert npd.check_partial_x(self._dx)
                if 'dy' in d_check_list:
                    assert npd.check_partial_y(self._dy)
                if 'dz' in d_check_list:
                    assert npd.check_partial_z(self._dz)

        # 4.2 do the checking: second derivative checking -------------------------------
        if len(dd_check_list) == 0:
            pass
        else:
            NPD_f = NumericalPartialDerivativeTxyzFunctions(self._s_)
            if 'dd_tt' in dd_check_list:
                dt_f = NPD_f('t')
                for i in X:
                    t = self._find_random_testing_time_instance_()
                    npd = NumericalPartialDerivativeTxyz(dt_f, t, X[i], Y[i], Z[i])
                    assert npd.check_partial_t(self._dd_tt, tolerance=1e-3)

            if 'dd_xx' in dd_check_list or 'dd_xy' in dd_check_list or 'dd_xz' in dd_check_list:
                dx_f = NPD_f('x')
                for i in X:
                    t = self._find_random_testing_time_instance_()
                    npd = NumericalPartialDerivativeTxyz(dx_f, t, X[i], Y[i], Z[i])
                    if 'dd_xx' in dd_check_list:
                        assert npd.check_partial_x(self._dd_xx, tolerance=1e-3)
                    if 'dd_xy' in dd_check_list:
                        assert npd.check_partial_y(self._dd_xy, tolerance=1e-3)
                    if 'dd_xz' in dd_check_list:
                        assert npd.check_partial_z(self._dd_xz, tolerance=1e-3)

            if 'dd_yx' in dd_check_list or 'dd_yy' in dd_check_list or 'dd_yz' in dd_check_list:
                dy_f = NPD_f('y')
                for i in X:
                    t = self._find_random_testing_time_instance_()
                    npd = NumericalPartialDerivativeTxyz(dy_f, t, X[i], Y[i], Z[i])
                    if 'dd_yx' in dd_check_list:
                        assert npd.check_partial_x(self._dd_yx, tolerance=1e-3)
                    if 'dd_yy' in dd_check_list:
                        assert npd.check_partial_y(self._dd_yy, tolerance=1e-3)
                    if 'dd_yz' in dd_check_list:
                        assert npd.check_partial_z(self._dd_yz, tolerance=1e-3)

            if 'dd_zx' in dd_check_list or 'dd_zy' in dd_check_list or 'dd_zz' in dd_check_list:
                dz_f = NPD_f('z')
                for i in X:
                    t = self._find_random_testing_time_instance_()
                    npd = NumericalPartialDerivativeTxyz(dz_f, t, X[i], Y[i], Z[i])
                    if 'dd_zx' in dd_check_list:
                        assert npd.check_partial_x(self._dd_zx, tolerance=1e-3)
                    if 'dd_zy' in dd_check_list:
                        assert npd.check_partial_y(self._dd_zy, tolerance=1e-3)
                    if 'dd_zz' in dd_check_list:
                        assert npd.check_partial_z(self._dd_zz, tolerance=1e-3)

        # =========================================================================
        self._mesh = _mesh

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
        from phyem.tools.functions.time_space._3d.wrappers.vector import T3dVector
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

    @property
    def Laplacian(self):
        r"""Return a T2dScalar instance representing laplace of self."""
        if self._dd_xx is None or self._dd_yy is None or self._dd_zz is None:
            u = self.gradient
            return u.divergence
        else:
            return self.__class__(t3d_ScalarAdd3(self._dd_xx, self._dd_yy, self._dd_zz), mesh=self.mesh)

    def log(self, base=np.e):
        r"""return a scalar function of (t, x, y) which computes log_{base} self(t, x, t).

        Be default, we compute log_e.

        """
        if base == np.e:
            if self._log_e_ is None:
                self._log_e_ = self.__class__(___LOG_HELPER_3___(self._s_, base=np.e), mesh=self.mesh)
            return self._log_e_
        else:
            raise NotImplementedError()

    @property
    def exp(self):
        r"""return a scalar function of (t, x, y) which computes exp^{self(t, x, y)}."""
        if self._exp_ is None:
            self._exp_ = self.__class__(___EXP_HELPER_3___(self._s_), mesh=self.mesh)
        return self._exp_

    @property
    def abs(self):
        r""""""
        return self.__class__(t3d_ScalarAbs(self._s_), mesh=self.mesh)

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

        elif isinstance(other, (int, float)):
            s0_mul_s1 = t3d_ScalarMultiply(self._s_, other)
            return self.__class__(s0_mul_s1)

        elif other.__class__.__name__ == 'T3dVector':
            s0v0 = t3d_ScalarMultiply(self._s_, other._v0_)
            s0v1 = t3d_ScalarMultiply(self._s_, other._v1_)
            s0v2 = t3d_ScalarMultiply(self._s_, other._v2_)
            return other.__class__(s0v0, s0v1, s0v2)

        else:
            raise NotImplementedError()

    def __rmul__(self, other):
        """other * self"""
        if isinstance(other, (int, float)):
            return self * other
        elif other.__class__.__name__ == 'T3dVector':
            return self * other
        else:
            raise NotImplementedError(other)
