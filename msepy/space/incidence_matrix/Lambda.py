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
from scipy.sparse import csr_matrix


class MsePyIncidenceMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        assert self._k != self._n, f"top form has no incidence matrix."
        self._orientation = space.abstract.orientation
        self._freeze()

    def __call__(self, degree):
        """Making the local numbering for degree."""
        assert isinstance(degree, (int, float)) and degree % 1 == 0 and degree > 0, f"degree wrong."
        if self._n == 2:
            return getattr(self, f"_n{self._n}_k{self._k}_{self._orientation}")(degree)
        else:
            return getattr(self, f"_n{self._n}_k{self._k}")(degree)

    def _n3_k2(self, p):
        """div or d of 2-form"""
        sn = self._space.local_numbering.Lambda._n3_k2(p)
        dn = self._space.local_numbering.Lambda._n3_k3(p)
        E = np.zeros((p**3, 3*(p+1)*p**2), dtype=int)

        I, J, K = np.shape(dn[0])
        for k in range(K):
            for j in range(J):
                for i in range(I):
                    E[dn[0][i, j, k], sn[0][i, j, k]] = -1    # x-
                    E[dn[0][i, j, k], sn[0][i+1, j, k]] = +1  # x+
                    E[dn[0][i, j, k], sn[1][i, j, k]] = -1    # y-
                    E[dn[0][i, j, k], sn[1][i, j+1, k]] = +1  # y+
                    E[dn[0][i, j, k], sn[2][i, j, k]] = -1    # z-
                    E[dn[0][i, j, k], sn[2][i, j, k+1]] = +1  # z+
        return csr_matrix(E)

    def _n3_k1(self, p):
        """curl or d of 1-form"""
        sn = self._space.local_numbering.Lambda._n3_k1(p)
        dn = self._space.local_numbering.Lambda._n3_k2(p)
        E = np.zeros((3*(p+1)*p**2, 3*p*(p+1)**2), dtype=int)

        I, J, K = np.shape(dn[0])
        for k in range(K):
            for j in range(J):
                for i in range(I):
                    E[dn[0][i, j, k], sn[1][i, j, k]] = +1   # Back
                    E[dn[0][i, j, k], sn[1][i, j, k+1]] = -1   # Front
                    E[dn[0][i, j, k], sn[2][i, j, k]] = -1   # West
                    E[dn[0][i, j, k], sn[2][i, j+1, k]] = +1   # East

        I, J, K = np.shape(dn[1])
        for k in range(K):
            for j in range(J):
                for i in range(I):
                    E[dn[1][i, j, k], sn[0][i, j, k]] = -1    # Back
                    E[dn[1][i, j, k], sn[0][i, j, k+1]] = +1    # Front
                    E[dn[1][i, j, k], sn[2][i, j, k]] = +1    # North
                    E[dn[1][i, j, k], sn[2][i+1, j, k]] = -1    # South

        I, J, K = np.shape(dn[2])
        for k in range(K):
            for j in range(J):
                for i in range(I):
                    E[dn[2][i, j, k], sn[0][i, j, k]] = +1    # West
                    E[dn[2][i, j, k], sn[0][i, j+1, k]] = -1    # East
                    E[dn[2][i, j, k], sn[1][i, j, k]] = -1    # North
                    E[dn[2][i, j, k], sn[1][i+1, j, k]] = +1    # South

        return csr_matrix(E)

    def _n3_k0(self, p):
        """grad or d of 0-form"""
        sn = self._space.local_numbering.Lambda._n3_k0(p)
        dn = self._space.local_numbering.Lambda._n3_k1(p)
        E = np.zeros((3*p*(p+1)**2, (p+1)**3), dtype=int)

        I, J, K = np.shape(dn[0])
        for k in range(K):
            for j in range(J):
                for i in range(I):
                    E[dn[0][i, j, k], sn[0][i, j, k]] = -1   # North
                    E[dn[0][i, j, k], sn[0][i+1, j, k]] = +1   # South

        I, J, K = np.shape(dn[1])
        for k in range(K):
            for j in range(J):
                for i in range(I):
                    E[dn[1][i, j, k], sn[0][i, j, k]] = -1    # West
                    E[dn[1][i, j, k], sn[0][i, j+1, k]] = +1    # East

        I, J, K = np.shape(dn[2])
        for k in range(K):
            for j in range(J):
                for i in range(I):
                    E[dn[2][i, j, k], sn[0][i, j, k]] = -1    # Back
                    E[dn[2][i, j, k], sn[0][i, j, k+1]] = +1    # Front

        return csr_matrix(E)

    def _n2_k0_inner(self, p):
        """grad or d of inner 0-form"""
        sn = self._space.local_numbering.Lambda._n2_k0(p)
        dn = self._space.local_numbering.Lambda._n2_k1_inner(p)
        E = np.zeros((2*p*(p+1), (p+1)**2), dtype=int)
        I, J = np.shape(dn[0])  # dx edges
        for j in range(J):
            for i in range(I):
                E[dn[0][i, j], sn[0][i, j]] = -1     # x-
                E[dn[0][i, j], sn[0][i+1, j]] = +1   # x+
        I, J = np.shape(dn[1])  # dy edges
        for j in range(J):
            for i in range(I):
                E[dn[1][i, j], sn[0][i, j]] = -1      # y-
                E[dn[1][i, j], sn[0][i, j+1]] = +1    # y+
        return csr_matrix(E)

    def _n2_k1_inner(self, p):
        """rot or d of inner 1-form"""
        sn = self._space.local_numbering.Lambda._n2_k1_inner(p)
        dn = self._space.local_numbering.Lambda._n2_k2(p)
        E = np.zeros((p**2,
                      2*p*(p+1)), dtype=int)
        I, J = np.shape(dn[0])
        for j in range(J):
            for i in range(I):
                E[dn[0][i, j], sn[1][i, j]] = -1      # x-
                E[dn[0][i, j], sn[1][i+1, j]] = +1    # x+
                E[dn[0][i, j], sn[0][i, j]] = +1      # y-
                E[dn[0][i, j], sn[0][i, j+1]] = -1    # y+
        return csr_matrix(E)

    def _n2_k0_outer(self, p):
        """curl or d of outer-0-form"""
        sn = self._space.local_numbering.Lambda._n2_k0(p)
        dn = self._space.local_numbering.Lambda._n2_k1_outer(p)
        E = np.zeros((2*p*(p+1), (p+1)**2), dtype=int)
        I, J = np.shape(dn[0])    # dy edges
        for j in range(J):
            for i in range(I):
                E[dn[0][i, j], sn[0][i, j]] = -1     # y-
                E[dn[0][i, j], sn[0][i, j+1]] = +1   # y+
        I, J = np.shape(dn[1])    # dx edges
        for j in range(J):
            for i in range(I):
                E[dn[1][i, j], sn[0][i, j]] = +1     # x-
                E[dn[1][i, j], sn[0][i+1, j]] = -1   # x+
        return csr_matrix(E)

    def _n2_k1_outer(self, p):
        """div or d of outer-1-form"""
        sn = self._space.local_numbering.Lambda._n2_k1_outer(p)
        dn = self._space.local_numbering.Lambda._n2_k2(p)
        E = np.zeros((p**2,
                      2*p*(p+1)), dtype=int)
        I, J = np.shape(dn[0])
        for j in range(J):
            for i in range(I):
                E[dn[0][i, j], sn[0][i, j]] = -1      # x-
                E[dn[0][i, j], sn[0][i+1, j]] = +1    # x+
                E[dn[0][i, j], sn[1][i, j]] = -1      # y-
                E[dn[0][i, j], sn[1][i, j+1]] = +1    # y+
        return csr_matrix(E)

    @staticmethod
    def _n1_k0(p):
        """"""
        E = np.zeros((p, p+1), dtype=int)
        for i in range(p):
            E[i, i] = -1   # x-
            E[i, i+1] = 1  # x+
        return csr_matrix(E)
