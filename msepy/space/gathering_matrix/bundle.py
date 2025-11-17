# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.frozen import Frozen
from phyem.msepy.tools.gathering_matrix import RegularGatheringMatrix


class MsePyGatheringMatrixBundle(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._mesh = space.mesh
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._orientation = space.abstract.orientation
        self._cache = dict()
        self._freeze()

    def __call__(self, degree):
        """Making the local numbering for degree."""
        p = self._space[degree].p
        cache_key = str(p)  # only need p
        if cache_key in self._cache:
            gm = self._cache[cache_key]
        else:
            if self._n == 2 and self._k == 1:
                method_name = f"_n{self._n}_k{self._k}_{self._orientation}"
            else:
                method_name = f"_n{self._n}_k{self._k}"
            gm = getattr(self, method_name)(p)
            self._cache[cache_key] = gm

        return gm

    def _n3_k3(self, p):
        """"""
        total_num_elements = self._mesh.elements._num
        num_local_dofs = self._space.num_local_dofs.bundle._n3_k3(p)
        total_num_dofs = total_num_elements * num_local_dofs
        gm = np.arange(0, total_num_dofs).reshape(
            (self._mesh.elements._num, num_local_dofs), order='C',
        )
        return RegularGatheringMatrix(gm)

    def _n3_k2(self, p):
        """"""
        p0, p1, p2 = p
        x_gm = self._space.gathering_matrix.Lambda._n3_k2(p0)
        y_gm = self._space.gathering_matrix.Lambda._n3_k2(p1)
        z_gm = self._space.gathering_matrix.Lambda._n3_k2(p2)
        gm = RegularGatheringMatrix([x_gm, y_gm, z_gm])
        return gm

    def _n3_k1(self, p):
        """"""
        p0, p1, p2 = p
        x_gm = self._space.gathering_matrix.Lambda._n3_k1(p0)
        y_gm = self._space.gathering_matrix.Lambda._n3_k1(p1)
        z_gm = self._space.gathering_matrix.Lambda._n3_k1(p2)
        gm = RegularGatheringMatrix([x_gm, y_gm, z_gm])
        return gm

    def _n3_k0(self, p):
        """"""
        p0, p1, p2 = p
        x_gm = self._space.gathering_matrix.Lambda._n3_k0(p0)
        y_gm = self._space.gathering_matrix.Lambda._n3_k0(p1)
        z_gm = self._space.gathering_matrix.Lambda._n3_k0(p2)
        gm = RegularGatheringMatrix([x_gm, y_gm, z_gm])
        return gm

    def _n2_k2(self, p):
        """"""
        total_num_elements = self._mesh.elements._num
        num_local_dofs = self._space.num_local_dofs.bundle._n2_k2(p)
        total_num_dofs = total_num_elements * num_local_dofs
        gm = np.arange(0, total_num_dofs).reshape(
            (self._mesh.elements._num, num_local_dofs), order='C',
        )
        return RegularGatheringMatrix(gm)

    def _n2_k1_inner(self, p):
        """"""
        p0, p1 = p
        x_gm = self._space.gathering_matrix.Lambda._n2_k1_inner(p0)
        y_gm = self._space.gathering_matrix.Lambda._n2_k1_inner(p1)
        gm = RegularGatheringMatrix([x_gm, y_gm])
        return gm

    def _n2_k1_outer(self, p):
        """"""
        p0, p1 = p
        x_gm = self._space.gathering_matrix.Lambda._n2_k1_outer(p0)
        y_gm = self._space.gathering_matrix.Lambda._n2_k1_outer(p1)
        gm = RegularGatheringMatrix([x_gm, y_gm])
        return gm

    def _n2_k0(self, p):
        """"""
        p0, p1 = p
        x_gm = self._space.gathering_matrix.Lambda._n2_k0(p0)
        y_gm = self._space.gathering_matrix.Lambda._n2_k0(p1)
        gm = RegularGatheringMatrix([x_gm, y_gm])
        return gm

    def _n1_k0(self, p):
        """"""
        element_map = self._mesh.elements.map
        gm = - np.ones((self._mesh.elements._num, self._space.num_local_dofs.bundle._n1_k0(p)), dtype=int)
        current = 0
        p = p[0]
        for e, mp in enumerate(element_map):
            # number x- node
            x_m = mp[0]
            if x_m == -1 or x_m > e:  # x- side of element #e is a boundary or not numbered
                gm[e, 0] = current
                current += 1
            else:
                gm[e, 0] = gm[x_m, -1]
            # node intermediate nodes
            gm[e, 1:-1] = np.arange(current, current + p - 1)
            current += p - 1

            # number x+ node
            x_p = mp[-1]
            if x_p == -1 or x_p > e:
                gm[e, -1] = current
                current += 1
            else:
                gm[e, -1] = gm[x_p, 0]
        return RegularGatheringMatrix(gm)

    def _n1_k1(self, p):
        """"""
        total_num_elements = self._mesh.elements._num
        num_local_dofs = self._space.num_local_dofs.bundle._n1_k1(p)
        total_num_dofs = total_num_elements * num_local_dofs
        gm = np.arange(0, total_num_dofs).reshape(
            (self._mesh.elements._num, num_local_dofs), order='C',
        )
        return RegularGatheringMatrix(gm)
