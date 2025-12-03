# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.tools.functions.time_space._2d.wrappers.scalar import T2dScalar
from phyem.tools.functions.time_space._2d.wrappers.vector import T2dVector
from phyem.tools.functions.time_space._3d.wrappers.scalar import T3dScalar
from phyem.tools.functions.time_space._3d.wrappers.vector import T3dVector


class MseHtt_Form_Numeric_TimeSpaceProperties(Frozen):
    r""""""

    def __init__(self, f):
        r""""""
        self._f = f
        self._freeze()

    def __call__(self, t=None):
        r"""return this time-space property. Like self.components, but now we return for example a vector
        instead of several scalars.
        """
        dtype = self._f.numeric.dtype
        if dtype == '2d-scalar':
            return T2dScalar(
                _ComponentHelper_(self._f.numeric._interpolate_, 0, t=t), steady=True
            )
        elif dtype == '3d-scalar':
            return T3dScalar(
                _ComponentHelper_(self._f.numeric._interpolate_, 0, t=t), steady=True
            )
        elif dtype == '2d-vector':
            components = self.components(t=t)
            v0 = components[0]._s_
            v1 = components[1]._s_
            return T2dVector(v0, v1)
        elif dtype == '3d-vector':
            components = self.components(t=t)
            v0 = components[0]._s_
            v1 = components[1]._s_
            v2 = components[2]._s_
            return T3dVector(v0, v1, v2)
        else:
            raise NotImplementedError()

    def components(self, t=None):
        r"""Return time-space scalar functions for all components of the form.

        The time of ``f`` is ``t``. When ``t`` is None, we always use the newest time of ``f``'s cochain.
        So the functions will update automatically and the t in (t, x, y) of the time-space scalars
        does not have any effect.

        Else if t is a number, then we return a steady time-space function. As the case of t=None,
        when we call the function use (t, x, y), t of (t, x, y) will have no effect at all.

        Parameters
        ----------
        t

        Returns
        -------

        """
        components = list()
        dtype = self._f.numeric.dtype
        if t is None:
            if dtype == '2d-vector':
                for i in range(2):
                    components.append(
                        T2dScalar(_ComponentHelper_(self._f.numeric._interpolate_, i))
                    )
            elif dtype == '3d-vector':
                for i in range(3):
                    components.append(
                        T3dScalar(_ComponentHelper_(self._f.numeric._interpolate_, i))
                    )
            else:
                raise NotImplementedError()
        elif isinstance(t, (int, float)):
            if dtype == '2d-vector':
                for i in range(2):
                    components.append(
                        T2dScalar(_ComponentHelper_(self._f.numeric._interpolate_, i, t=t), steady=True)
                    )
            elif dtype == '3d-vector':
                for i in range(3):
                    components.append(
                        T3dScalar(_ComponentHelper_(self._f.numeric._interpolate_, i, t=t), steady=True)
                    )
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        return components

    def L2_energy(self, t=None):
        r"""Return a function e(t, x, y) gives the L2-energy, i.e. 0.5 * (f, f), at (t, x, y).

        The time of f is ``t``. When ``t`` is None, we always use the newest time of its cochain. So this energy
        will update automatically and the t in e(t, x, y) does not have any effect.

        """
        if t is None:
            ndim = self._f.space.n
            if ndim == 2:
                return T2dScalar(self.___newest_energy___)
            elif ndim == 3:
                return T3dScalar(self.___newest_energy___)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def ___newest_energy___(self, t, *xyz):
        r""""""
        _ = t  # t has no effect since we will always use the newest t of the cochain.
        dtype, V = self._f.numeric._interpolate_(component_wise=True)

        if dtype == '2d-scalar':
            return 0.5 * V[0](*xyz) ** 2

        elif dtype == '2d-vector':
            return 0.5 * (V[0](*xyz) ** 2 + V[1](*xyz) ** 2)

        elif dtype == '3d-scalar':
            return 0.5 * V[0](*xyz) ** 2

        elif dtype == '3d-vector':
            return 0.5 * (V[0](*xyz) ** 2 + V[1](*xyz) ** 2 + V[2](*xyz) ** 2)

        else:
            raise NotImplementedError()


class _ComponentHelper_(Frozen):
    r""""""

    def __init__(self, itp_func, ith_component, t=None):
        r""""""
        self._itp_func_ = itp_func
        self._ith_component = ith_component
        self._t = t
        self._freeze()

    def __call__(self, t, *xyz):
        r""""""
        if self._t is None:
            _ = t
            itp = self._itp_func_(t=None, component_wise=True)[1]  # [1] since [0] is the dtype.
            return itp[self._ith_component](*xyz)
        else:
            itp = self._itp_func_(t=self._t, component_wise=True)[1]  # [1] since [0] is the dtype.
            return itp[self._ith_component](*xyz)
