# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import numpy as np
from scipy.sparse import csr_matrix
from generic.py.matrix.localize.static import Localize_Static_Matrix


class IncidenceMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._k = space.abstract.k
        self._orientation = space.abstract.orientation
        self._cache = {}
        self._freeze()

    def __call__(self, degree):
        """"""
        p = self._space[degree].p
        if p in self._cache:
            E = self._cache[p]

        else:
            method_name = f"_k{self._k}_{self._orientation}"
            raw_E = getattr(self, method_name)(p)

            if self._k == 1:
                csm = self._space.basis_functions.csm(degree)
            elif self._k == 0:
                if self._orientation == 'inner':
                    csm = self._space.basis_functions._csm_Lambda_k1_inner(degree)
                elif self._orientation == 'outer':
                    csm = self._space.basis_functions._csm_Lambda_k1_outer(degree)
                else:
                    raise Exception
            else:
                raise Exception

            E = dict()
            for index in self._space.mesh:
                ele_type = self._space.mesh[index].type
                e = raw_E[ele_type]
                if index in csm:
                    if self._k == 1:
                        e = e @ csm[index]
                    elif self._k == 0:
                        e = csm[index] @ e
                    else:
                        raise Exception()
                else:
                    pass

                E[index] = e

            self._cache[p] = E

        gm_col = self._space.gathering_matrix(degree)
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            if self._k == 1:
                gm_row = self._space.gathering_matrix.Lambda._k2(degree)
            elif self._k == 0 and self._orientation == 'outer':
                gm_row = self._space.gathering_matrix.Lambda._k1_outer(degree)
            elif self._k == 0 and self._orientation == 'inner':
                gm_row = self._space.gathering_matrix.Lambda._k1_inner(degree)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        E = Localize_Static_Matrix(E, gm_row, gm_col)
        return E

    def _k0_outer(self, p):
        """curl or d of outer-0-form"""
        local_numbering_1 = self._space.local_numbering.Lambda._k1_outer(p)
        local_numbering_0 = self._space.local_numbering.Lambda._k0(p)
        num_local_dofs_1 = self._space.num_local_dofs.Lambda._k1(p)
        num_local_dofs_0 = self._space.num_local_dofs.Lambda._k0(p)

        sn = local_numbering_0['q']
        dn = local_numbering_1['q']
        E = np.zeros(
            (
                num_local_dofs_1['q'],
                num_local_dofs_0['q']
            ),
            dtype=int
        )
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
        _q_ = csr_matrix(E)

        sn = local_numbering_0['t']
        dn = local_numbering_1['t']
        E = np.zeros(
            (
                num_local_dofs_1['t'],
                num_local_dofs_0['t']
            ),
            dtype=int
        )
        I, J = np.shape(dn[0])
        for j in range(J):
            for i in range(I):
                E[dn[0][i, j], sn[0][i+1, j]] = -1     # y-
                E[dn[0][i, j], sn[0][i+1, j+1]] = +1   # y+
        I, J = np.shape(dn[1])
        for j in range(J):
            for i in range(I):
                E[dn[1][i, j], sn[0][i, j]] = +1       # x-
                E[dn[1][i, j], sn[0][i+1, j]] = -1     # x+
        _t_ = csr_matrix(E)

        return {
            'q': _q_,
            't': _t_,
        }

    def _k1_outer(self, p):
        """div or d of outer-1-form"""
        local_numbering_1 = self._space.local_numbering.Lambda._k1_outer(p)
        local_numbering_2 = self._space.local_numbering.Lambda._k2(p)
        num_local_dofs_2 = self._space.num_local_dofs.Lambda._k2(p)
        num_local_dofs_1 = self._space.num_local_dofs.Lambda._k1(p)

        sn = local_numbering_1['q']
        dn = local_numbering_2['q']
        E = np.zeros(
            (
                num_local_dofs_2['q'],
                num_local_dofs_1['q']
            ),
            dtype=int
        )
        I, J = np.shape(dn[0])
        for j in range(J):
            for i in range(I):
                E[dn[0][i, j], sn[0][i, j]] = -1      # x-
                E[dn[0][i, j], sn[0][i+1, j]] = +1    # x+
                E[dn[0][i, j], sn[1][i, j]] = -1      # y-
                E[dn[0][i, j], sn[1][i, j+1]] = +1    # y+
        _q_ = csr_matrix(E)

        sn = local_numbering_1['t']
        dn = local_numbering_2['t']
        E = np.zeros(
            (
                num_local_dofs_2['t'],
                num_local_dofs_1['t']
            ),
            dtype=int
        )
        I, J = np.shape(dn[0])
        for j in range(J):
            for i in range(I):
                if i > 0:
                    E[dn[0][i, j], sn[0][i-1, j]] = -1        # x-
                E[dn[0][i, j], sn[0][i, j]] = +1              # x+
                E[dn[0][i, j], sn[1][i, j]] = -1              # y-
                E[dn[0][i, j], sn[1][i, j+1]] = +1            # y+
        _t_ = csr_matrix(E)

        return {
            'q': _q_,
            't': _t_,
        }

    def _k0_inner(self, p):
        """grad or d of inner 0-form"""
        local_numbering_0 = self._space.local_numbering.Lambda._k0(p)
        local_numbering_1 = self._space.local_numbering.Lambda._k1_inner(p)
        num_local_dofs_1 = self._space.num_local_dofs.Lambda._k1(p)
        num_local_dofs_0 = self._space.num_local_dofs.Lambda._k0(p)

        # --------- quadrilateral cell -----------------------------------------------
        sn = local_numbering_0['q']
        dn = local_numbering_1['q']
        E = np.zeros(
            (
                num_local_dofs_1['q'],
                num_local_dofs_0['q']
            ),
            dtype=int
        )
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
        _q_ = csr_matrix(E)

        sn = local_numbering_0['t']
        dn = local_numbering_1['t']
        E = np.zeros(
            (
                num_local_dofs_1['t'],
                num_local_dofs_0['t']
            ),
            dtype=int
        )
        I, J = np.shape(dn[0])
        for j in range(J):           # dx edges
            for i in range(I):
                E[dn[0][i, j], sn[0][i, j]] = -1     # x-
                E[dn[0][i, j], sn[0][i+1, j]] = +1   # x+

        I, J = np.shape(dn[1])
        for j in range(J):           # dy edges
            for i in range(I):
                E[dn[1][i, j], sn[0][i+1, j]] = -1    # y-
                E[dn[1][i, j], sn[0][i+1, j+1]] = +1  # y+
        _t_ = csr_matrix(E)

        return {
            'q': _q_,
            't': _t_,
        }

    def _k1_inner(self, p):
        """rot or d of inner 1-form"""
        local_numbering_1 = self._space.local_numbering.Lambda._k1_inner(p)
        local_numbering_2 = self._space.local_numbering.Lambda._k2(p)
        num_local_dofs_2 = self._space.num_local_dofs.Lambda._k2(p)
        num_local_dofs_1 = self._space.num_local_dofs.Lambda._k1(p)

        sn = local_numbering_1['q']
        dn = local_numbering_2['q']
        E = np.zeros(
            (
                num_local_dofs_2['q'],
                num_local_dofs_1['q']
            ),
            dtype=int
        )
        I, J = np.shape(dn[0])
        for j in range(J):
            for i in range(I):
                E[dn[0][i, j], sn[1][i, j]] = -1      # x-
                E[dn[0][i, j], sn[1][i+1, j]] = +1    # x+
                E[dn[0][i, j], sn[0][i, j]] = +1      # y-
                E[dn[0][i, j], sn[0][i, j+1]] = -1    # y+
        _q_ = csr_matrix(E)

        sn = local_numbering_1['t']
        dn = local_numbering_2['t']
        E = np.zeros(
            (
                num_local_dofs_2['t'],
                num_local_dofs_1['t']
            ),
            dtype=int
        )
        I, J = np.shape(dn[0])
        for j in range(J):
            for i in range(I):
                if i > 0:
                    E[dn[0][i, j], sn[1][i-1, j]] = -1      # x-
                E[dn[0][i, j], sn[1][i, j]] = +1            # x+
                E[dn[0][i, j], sn[0][i, j]] = +1            # y-
                E[dn[0][i, j], sn[0][i, j+1]] = -1          # y+
        _t_ = csr_matrix(E)

        return {
            'q': _q_,
            't': _t_,
        }
