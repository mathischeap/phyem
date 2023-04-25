# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
import sys
if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
import numpy as np


class MsePyIncidenceMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        assert self._k != self._n, f"top form has no incidence matrix."
        self._orientation = space.abstract.orientation
        self._freeze()

    def __call__(self, degree):
        """Making the local numbering for degree."""
        assert isinstance(degree, (int, float)) and degree % 1 == 0 and degree > 0, f"degree wrong."
        if self._n == 2 and self._k == 1:
            return getattr(self, f"_n{self._n}_k{self._k}_{self._orientation}")(degree)
        else:
            return getattr(self, f"_n{self._n}_k{self._k}")(degree)

    @staticmethod
    def _n3_k2(p):
        """"""
        P = p*p*(p+1)
        # faces perp to x-axis
        local_numbering_dy_dz = np.arange(0 * P, 1 * P).reshape((p+1, p, p), order='F')
        # faces perp to y-axis
        local_numbering_dz_dx = np.arange(1 * P, 2 * P).reshape((p, p+1, p), order='F')
        # faces perp to z-axis
        local_numbering_dx_dy = np.arange(2 * P, 3 * P).reshape((p, p, p+1), order='F')
        return local_numbering_dy_dz, local_numbering_dz_dx, local_numbering_dx_dy

    @staticmethod
    def _n3_k1(p):
        """"""
        P = p * (p+1) * (p+1)
        local_numbering_dx = np.arange(0 * P, 1 * P).reshape((p, p+1, p+1), order='F')
        local_numbering_dy = np.arange(1 * P, 2 * P).reshape((p+1, p, p+1), order='F')
        local_numbering_dz = np.arange(2 * P, 3 * P).reshape((p+1, p+1, p), order='F')
        return local_numbering_dx, local_numbering_dy, local_numbering_dz

    @staticmethod
    def _n3_k0(p):
        """"""
        local_numbering = np.arange(0, (p+1)**3).reshape((p+1, p+1, p+1), order='F')
        return (local_numbering,)  # do not remove (,)

    @staticmethod
    def _n2_k0(p):
        """"""
        local_numbering = np.arange(0, (p+1)**2).reshape((p+1, p+1), order='F')
        return (local_numbering,)  # do not remove (,)

    @staticmethod
    def _n2_k1_outer(p):
        """"""
        P = p * (p+1)
        # segments perp to x-axis
        local_numbering_dy = np.arange(0 * P, 1 * P).reshape((p+1, p), order='F')
        # segments perp to y-axis
        local_numbering_dx = np.arange(1 * P, 2 * P).reshape((p, p+1), order='F')
        return local_numbering_dy, local_numbering_dx

    @staticmethod
    def _n2_k1_inner(p):
        """"""
        P = p * (p+1)
        local_numbering_dx = np.arange(0 * P, 1 * P).reshape((p, p+1), order='F')
        local_numbering_dy = np.arange(1 * P, 2 * P).reshape((p+1, p), order='F')
        return local_numbering_dx, local_numbering_dy

    @staticmethod
    def _n1_k0(p):
        """"""
        local_numbering = np.arange(0, p+1)
        return (local_numbering,)  # do not remove (,)
