# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import numpy as np
from scipy.sparse import csr_matrix


class _IncidenceMatrixWrapper(Frozen):
    """"""
    def __init__(self, elements, E, csm, axis):
        self._elements = elements
        self._E = E
        self._csm = csm
        self._axis = axis   # which axis of E csm is working on.
        self._freeze()

    @staticmethod
    def _is_dict_like():
        """A signature."""
        return True

    def __getitem__(self, index):
        """"""
        if isinstance(index, str):
            E = self._E['t']
        else:
            E = self._E['q']

        if index not in self._csm:
            return E
        else:
            total_csm = self._csm[index]
            if self._axis == 0:
                return total_csm @ E
            elif self._axis == 1:
                return E @ total_csm
            else:
                raise Exception

    def __iter__(self):
        """"""
        for i in self._elements:
            yield i

    def __len__(self):
        return len(self._elements)


class MseHyPy2IncidenceMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._orientation = space.abstract.orientation
        self._cache = dict()
        self._freeze()

    def __call__(self, degree, g):
        """"""
        p = self._space[degree].p

        key = f"{p}"

        if key in self._cache:
            E = self._cache[key]
        else:
            method_name = f"_k{self._k}_{self._orientation}"
            E = getattr(self, method_name)(p)
            self._cache[key] = E

        csm = self._space.basis_functions._cochain_switch_matrix_Lambda_k1(
            degree, g, self._orientation)[1]

        if self._k == 0:
            E = _IncidenceMatrixWrapper(self._space.mesh[g], E, csm, axis=0)
        elif self._k == 1:
            E = _IncidenceMatrixWrapper(self._space.mesh[g], E, csm, axis=1)
        else:
            raise Exception()

        return E

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

    def _k0_outer(self, p):
        """curl or d of outer-0-form"""
        local_numbering_0 = self._space.local_numbering.Lambda._k0(p)
        local_numbering_1 = self._space.local_numbering.Lambda._k1_outer(p)
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
