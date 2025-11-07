# -*- coding: utf-8 -*-
r"""
"""
from src.config import MASTER_RANK, RANK
from tools.frozen import Frozen
from tools.functions.time_space._3d.wrappers.scalar import T3dScalar
from tools.functions.time_space._3d.wrappers.vector import T3dVector

import pickle


class MseHtt_Static_Form_Export(Frozen):
    r""""""

    def __init__(self, f, t):
        r""""""
        self._t = t
        self._f = f
        self._rws = None
        self._freeze()

    def rws(self, filename, ddf=1):
        r""""""
        dds = self._f.numeric.rws(self._t, ddf=ddf)
        if dds is None:
            pass
        else:
            dds.saveto(filename)

    def vtk(self, filename, ddf=1):
        r""""""
        from tools.vtk_.msehtt_form_static_copy import ___ph_vtk_msehtt_static_copy___
        ___ph_vtk_msehtt_static_copy___(filename, self._f[self._t], ddf=ddf)

    def tsf(self, filename, ddf=1):
        r"""We export this form at time `self._t` as a time-space function, like a time-space scalar,
        vector or tensor or else.

        These time-space functions are not suitable for computing their derivatives.
        """
        dtype, itp = self._f.numeric._interpolate_global_(t=self._t, ddf=ddf, component_wise=True)
        if RANK != MASTER_RANK:
            return
        else:
            pass

        if dtype == '3d-scalar':
            P = itp[0]
            to_pk = T3dScalar(___FUNCTION_TIME_WRAPPER___(P), steady=True)

        elif dtype == '3d-vector':
            U, V, W = itp
            to_pk = T3dVector(
                ___FUNCTION_TIME_WRAPPER___(U),
                ___FUNCTION_TIME_WRAPPER___(V),
                ___FUNCTION_TIME_WRAPPER___(W),
                steady=True
            )

        else:
            raise NotImplementedError(f"export.tsf not implemented for dtype={dtype}.")

        with open(filename, 'wb') as output:
            # noinspection PyTypeChecker
            pickle.dump(to_pk, output, pickle.HIGHEST_PROTOCOL)
        output.close()

    def tsp(self, property_of_what, filename, ddf=1, **kwargs):
        r"""We export a property of this form as a time-space function,
        like a time-space scalar, vector or tensor or else.

        Parameters
        ----------
        property_of_what
        filename
        ddf
        kwargs :
            According to `property_of_what`, we may need different key-word arguments.

        Returns
        -------

        """
        raise NotImplementedError()


class ___FUNCTION_TIME_WRAPPER___:
    r""""""
    def __init__(self, func_without_t_input):
        r""""""
        self._func_without_t_input = func_without_t_input

    def __call__(self, t, *coordinates):
        r""""""
        return self._func_without_t_input(*coordinates)
