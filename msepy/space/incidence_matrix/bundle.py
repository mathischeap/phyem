# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import csr_matrix, bmat

from phyem.tools.frozen import Frozen


class MsePyIncidenceMatrixBundle(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._orientation = space.abstract.orientation
        self._cache = dict()
        self._freeze()

    def __call__(self, degree):
        """Making the local numbering for degree."""
        p = self._space[degree].p

        key = f"{p}"

        if key in self._cache:
            E = self._cache[key]
        else:
            if self._n == 2:  # for k == 0 and k == 1.
                method_name = f"_n{self._n}_k{self._k}_{self._orientation}"
            else:
                method_name = f"_n{self._n}_k{self._k}"
            E = getattr(self, method_name)(p)
            self._cache[key] = E

        return E

    def _n1_k0(self, p):
        """"""
        E = np.zeros(
            (
                self._space.num_local_dofs.Lambda._n1_k1(p),
                self._space.num_local_dofs.Lambda._n1_k0(p)
            ),
            dtype=int
        )
        for i in range(p[0]):
            E[i, i] = -1   # x-
            E[i, i+1] = 1  # x+
        return csr_matrix(E)

    def _n3_k0(self, p):
        """"""
        p0, p1, p2 = p
        E00 = self._space.incidence_matrix.Lambda._n3_k0(p0)
        E11 = self._space.incidence_matrix.Lambda._n3_k0(p1)
        E22 = self._space.incidence_matrix.Lambda._n3_k0(p2)

        return bmat([
            [E00, None, None],
            [None, E11, None],
            [None, None, E22]
        ])

    def _n3_k1(self, p):
        """"""
        p0, p1, p2 = p
        E00 = self._space.incidence_matrix.Lambda._n3_k1(p0)
        E11 = self._space.incidence_matrix.Lambda._n3_k1(p1)
        E22 = self._space.incidence_matrix.Lambda._n3_k1(p2)

        return bmat([
            [E00, None, None],
            [None, E11, None],
            [None, None, E22]
        ])

    def _n3_k2(self, p):
        """"""
        p0, p1, p2 = p
        E00 = self._space.incidence_matrix.Lambda._n3_k2(p0)
        E11 = self._space.incidence_matrix.Lambda._n3_k2(p1)
        E22 = self._space.incidence_matrix.Lambda._n3_k2(p2)

        return bmat([
            [E00, None, None],
            [None, E11, None],
            [None, None, E22]
        ])

    def _n2_k0_outer(self, p):
        """"""
        p0, p1 = p
        E00 = self._space.incidence_matrix.Lambda._n2_k0_outer(p0)
        E11 = self._space.incidence_matrix.Lambda._n2_k0_outer(p1)

        return bmat([
            [E00, None],
            [None, E11],
        ])

    def _n2_k0_inner(self, p):
        """"""
        p0, p1 = p

        E00 = self._space.incidence_matrix.Lambda._n2_k0_inner(p0)
        E11 = self._space.incidence_matrix.Lambda._n2_k0_inner(p1)

        return bmat([
            [E00, None],
            [None, E11],
        ])

    def _n2_k1_outer(self, p):
        """"""
        p0, p1 = p
        E00 = self._space.incidence_matrix.Lambda._n2_k1_outer(p0)
        E11 = self._space.incidence_matrix.Lambda._n2_k1_outer(p1)

        return bmat([
            [E00, None],
            [None, E11],
        ])

    def _n2_k1_inner(self, p):
        """"""
        p0, p1 = p
        E00 = self._space.incidence_matrix.Lambda._n2_k1_inner(p0)
        E11 = self._space.incidence_matrix.Lambda._n2_k1_inner(p1)

        return bmat([
            [E00, None],
            [None, E11],
        ])
