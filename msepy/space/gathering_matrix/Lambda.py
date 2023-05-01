# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
import sys
if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.tools.gathering_matrix import RegularGatheringMatrix
import numpy as np


class MsePyGatheringMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._mesh = space.mesh
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._cache = dict()
        self._freeze()

    def __call__(self, degree):
        """Making the local numbering for degree."""
        p = self._space[degree].p
        method_name = f"_n{self._n}_k{self._k}"
        cache_key = method_name + ':' + str(p)
        if cache_key not in self._cache:
            self._cache[cache_key] = getattr(self, method_name)(p)
        else:
            pass
        return self._cache[cache_key]

    @staticmethod
    def _n3_k3(p):
        """"""
        raise NotImplementedError

    @staticmethod
    def _n3_k2(p):
        """"""
        raise NotImplementedError

    @staticmethod
    def _n3_k1(p):
        """"""
        raise NotImplementedError

    @staticmethod
    def _n3_k0(p):
        """"""
        raise NotImplementedError

    @staticmethod
    def _n2_k0(p):
        """"""
        raise NotImplementedError

    @staticmethod
    def _n2_k1(p):
        """"""
        raise NotImplementedError

    @staticmethod
    def _n2_k2(p):
        """"""
        raise NotImplementedError

    def _n1_k0(self, p):
        """"""
        element_map = self._mesh.elements.map
        gm = - np.ones((self._mesh.elements._num, self._space.num_local_dofs.Lambda._n1_k0(p)), dtype=int)
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
        element_map = self._mesh.elements.map
        gm = - np.ones((self._mesh.elements._num, self._space.num_local_dofs.Lambda._n1_k1(p)), dtype=int)
        current = 0
        p = p[0]
        for e, mp in enumerate(element_map):
            gm[e, :] = np.arange(current, current + p)
            current += p

        return RegularGatheringMatrix(gm)
